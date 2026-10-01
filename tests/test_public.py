import unittest
from datetime import datetime, timezone, timedelta, time
from decimal import Decimal

from rampflow.timeutil import Window
from rampflow.event_stream import EventStream
from rampflow.ops_store import VersionedStore, VersionConflict
from rampflow.operational_snapshot import SnapshotCodec
from rampflow.fuel_accounting import FuelTank, FuelLedger
from rampflow.crew_duty_control import CrewMember, CrewDutyController, DutySegment
from rampflow.operational_policy import PolicyRule, PolicyContext, PolicyEngine, Effect
from rampflow.service_windows import ServiceWindowRule, ServiceWindowPlanner
from rampflow.resource_dispatch import TravelGraph, MobileResource, ResourceDispatcher, DispatchRequest, DispatchError
from rampflow.stand_orchestration import StandProfile, StandDemand, StandOrchestrator
from rampflow.turnaround import TurnaroundPlanner, TurnTask, ResourceRequirement
from rampflow.dispatch import Dispatcher, DispatchUnit, DispatchJob, DispatchAssignment
from rampflow.models import TaskSpec

UTC=timezone.utc
D=lambda h,m=0: datetime(2026,1,1,h,m,tzinfo=UTC)

class PublicTests(unittest.TestCase):
    def test_event_stream_idempotent_publish(self):
        s=EventStream()
        a=s.publish('e1',D(10),'ops','F1',{'x':1})
        b=s.publish('e1',D(10),'ops','F1',{'x':1})
        self.assertEqual(a,b)
        self.assertTrue(s.verify_chain())

    def test_versioned_store_compare_and_swap(self):
        store=VersionedStore()
        rec=store.create('x',{'v':1},'a','create',D(10))
        newer=store.put('x',{'v':2},rec.version,'a','change',D(11))
        self.assertEqual(newer.version,2)
        with self.assertRaises(VersionConflict):
            store.put('x',{'v':3},1,'a','stale',D(12))

    def test_snapshot_round_trip(self):
        codec=SnapshotCodec()
        snap=codec.create('s',2,{'b':2,'a':1},D(10))
        self.assertEqual(codec.decode(codec.encode(snap)).digest,snap.digest)

    def test_fuel_delivery_reduces_inventory(self):
        ledger=FuelLedger([FuelTank('T1',Decimal('10000'),Decimal('5000'),Decimal('0.8'))])
        d=ledger.record('D1','F1','T1',Decimal('10'),Decimal('1010'),Decimal('15'),Decimal('0.8'),Decimal('1.2'),D(10))
        self.assertEqual(d.corrected_l,Decimal('1000.000'))
        self.assertEqual(ledger.inventory_l['T1'],Decimal('4000.000'))

    def test_crew_qualification_enforced(self):
        c=CrewDutyController([CrewMember('C1',frozenset({'push'}),'A',600,600,240)])
        ok,_=c.can_accept('C1',DutySegment('s',Window(D(10),D(11)),'A','push'))
        bad,_=c.can_accept('C1',DutySegment('s2',Window(D(11),D(12)),'A','fuel'))
        self.assertTrue(ok);self.assertFalse(bad)

    def test_policy_requirement(self):
        r=PolicyRule('R',10,frozenset({'push'}),Effect.REQUIRE,'approval required')
        e=PolicyEngine([r])
        ctx=PolicyContext('push','F','APT','T1',D(10),{})
        self.assertFalse(e.evaluate(ctx).allowed)
        ctx2=PolicyContext('push','F','APT','T1',D(10),{'requirement:R':True})
        self.assertTrue(e.evaluate(ctx2).allowed)

    def test_service_window_dependency(self):
        p=ServiceWindowPlanner([
            ServiceWindowRule('clean',0,60,10),
            ServiceWindowRule('board',0,90,20,prerequisites=frozenset({'clean'})),
        ])
        plan=p.plan('F',D(10),D(12),{})
        self.assertLessEqual(plan.bookings['clean'].window.end,plan.bookings['board'].window.start)

    def test_dispatch_shortest_path(self):
        g=TravelGraph();g.add('A','B',100);g.add('B','C',100);g.add('A','C',500)
        d=ResourceDispatcher([MobileResource('R','bus',frozenset(),'A',D(9),10)],g)
        req=DispatchRequest('Q','bus',None,'C',D(10),5,D(11))
        plan=d.plan(req)
        self.assertEqual((plan.arrive_at-plan.depart_at).total_seconds(),20)

    def test_stand_capacity(self):
        o=StandOrchestrator([StandProfile('S','T',40,True,1)])
        a=o.plan(StandDemand('F',Window(D(10),D(11)),35,False))
        o.commit(a)
        with self.assertRaises(Exception):
            o.plan(StandDemand('G',Window(D(10,30),D(11,30)),35,False))

    def test_dispatch_waits_for_resource_readiness(self):
        g=TravelGraph();g.add('A','B',600)
        d=ResourceDispatcher([MobileResource('R','bus',frozenset(),'A',D(10,30),10)],g)
        plan=d.plan(DispatchRequest('Q','bus',None,'B',D(10),5,D(11)))
        self.assertGreaterEqual(plan.depart_at,D(10,30))
        d.commit(plan)
        self.assertEqual(d.state('R').location,'B')

    def test_dispatch_race_loser_rejected_without_residue(self):
        g=TravelGraph();g.add('A','B',600);g.add('A','C',600)
        d=ResourceDispatcher([MobileResource('R','bus',frozenset(),'A',D(9),10)],g)
        p1=d.plan(DispatchRequest('Q1','bus',None,'B',D(10),5,D(11)))
        p2=d.plan(DispatchRequest('Q2','bus',None,'C',D(10),5,D(11)))
        d.commit(p1)
        with self.assertRaises(DispatchError):
            d.commit(p2)
        self.assertEqual(d.state('R').location,'B')
        self.assertNotIn('Q2',d.dispatches)

    def test_turnaround_stale_plan_conflict_rejected(self):
        p=TurnaroundPlanner({'bus':['V1']})
        task=TurnTask(TaskSpec('tow',30),requirements=(ResourceRequirement('bus'),))
        plan1=p.plan('F1',[task],D(10),D(12))
        plan2=p.plan('F2',[task],D(10),D(12))
        self.assertTrue(p.commit(plan1).accepted)
        denied=p.commit(plan2)
        self.assertFalse(denied.accepted)
        self.assertNotIn('F2',p.plans)
        self.assertEqual(len(p.ledger.reservations('V1')),1)

    def test_turnaround_stale_plan_revalidated_when_fits(self):
        p=TurnaroundPlanner({'bus':['V1']})
        early=TurnTask(TaskSpec('tow',30),requirements=(ResourceRequirement('bus'),))
        late=TurnTask(TaskSpec('load',30,earliest_offset_min=60),requirements=(ResourceRequirement('bus'),))
        plan1=p.plan('F1',[early],D(10),D(12))
        plan2=p.plan('F2',[late],D(10),D(12))
        self.assertTrue(p.commit(plan1).accepted)
        reconciled=p.commit(plan2)
        self.assertTrue(reconciled.accepted)
        self.assertEqual(reconciled.code,'reconciled')
        self.assertEqual(len(p.ledger.reservations('V1')),2)

    def _unit_dispatcher(self):
        u=DispatchUnit('U','bus','A',frozenset(),30.0,Window(D(8),D(20)))
        return Dispatcher([u],lambda a,b:0.0)

    def test_commit_batch_rejects_intra_batch_overlap(self):
        d=self._unit_dispatcher()
        j1=DispatchJob('J1','bus','A',D(10),D(12),30)
        j2=DispatchJob('J2','bus','A',D(10),D(12),30)
        a1=DispatchAssignment('J1','U',None,Window(D(10),D(10,30)),0.0,1.0)
        a2=DispatchAssignment('J2','U',None,Window(D(10,15),D(10,45)),0.0,1.0)
        dec=d.commit_batch([a1,a2],{'J1':j1,'J2':j2})
        self.assertFalse(dec.accepted)
        self.assertEqual(d.assignments,{})
        self.assertEqual(d._use['U'],[])

    def test_commit_batch_unknown_job_leaves_no_occupation(self):
        d=self._unit_dispatcher()
        j1=DispatchJob('J1','bus','A',D(10),D(12),30)
        a1=DispatchAssignment('J1','U',None,Window(D(10),D(10,30)),0.0,1.0)
        a2=DispatchAssignment('JX','U',None,Window(D(11),D(11,30)),0.0,1.0)
        dec=d.commit_batch([a1,a2],{'J1':j1})
        self.assertFalse(dec.accepted)
        self.assertEqual(d.assignments,{})
        self.assertEqual(d._use['U'],[])

if __name__=='__main__': unittest.main()
