import unittest
from datetime import datetime, timezone

from rampflow.ledger import ConflictError, ReservationLedger
from rampflow.models import ServiceState, TaskSpec
from rampflow.timeutil import Window
from rampflow.turnaround import ResourceRequirement, TurnTask, TurnaroundPlanner
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc


def D(hour, minute=0):
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def loading_task(task_id="load", duration=30, offset=0, predecessors=(), vehicle_kind="belt"):
    spec = TaskSpec(task_id, duration, frozenset(predecessors), earliest_offset_min=offset)
    return TurnTask(spec, requirements=(ResourceRequirement(vehicle_kind),))


class TurnaroundContentionTests(unittest.TestCase):
    def test_stale_plan_cannot_bypass_reservation(self):
        planner = TurnaroundPlanner({"belt": ("veh-1",)})
        plan_a = planner.plan("FA", [loading_task()], D(10), D(12))
        plan_b = planner.plan("FB", [loading_task()], D(10), D(12))
        self.assertEqual(plan_a.resource_generation, plan_b.resource_generation)

        self.assertTrue(planner.commit(plan_a).accepted)
        decision = planner.commit(plan_b)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "stale-generation")
        self.assertNotIn("FB", planner.plans)
        held = planner.ledger.reservations("veh-1")
        self.assertEqual(len(held), 1)
        self.assertTrue(all(r.payload.startswith("FA:") for r in held))

    def test_deterministic_reassignment_after_replan(self):
        planner = TurnaroundPlanner({"belt": ("veh-1", "veh-2")})
        plan_a = planner.plan("FA", [loading_task()], D(10), D(12))
        stale_b = planner.plan("FB", [loading_task()], D(10), D(12))
        planner.commit(plan_a)
        self.assertFalse(planner.commit(stale_b).accepted)

        replanned = planner.plan("FB", [loading_task()], D(10), D(12))
        self.assertEqual(replanned.runs["load"].resources, ("veh-2",))
        self.assertEqual(replanned.runs["load"].planned, Window(D(10), D(10, 30)))
        self.assertTrue(planner.commit(replanned).accepted)

    def test_single_vehicle_replan_shifts_window_deterministically(self):
        planner = TurnaroundPlanner({"belt": ("veh-1",)})
        planner.commit(planner.plan("FA", [loading_task()], D(10), D(12)))

        replanned = planner.plan("FB", [loading_task()], D(10), D(12))
        self.assertEqual(replanned.runs["load"].planned, Window(D(10, 30), D(11)))
        self.assertTrue(planner.commit(replanned).accepted)
        held = planner.ledger.reservations("veh-1")
        windows = sorted((r.window.start, r.window.end) for r in held)
        self.assertEqual(windows, [(D(10), D(10, 30)), (D(10, 30), D(11))])

    def test_failed_reservation_leaves_no_equipment_held(self):
        ledger = ReservationLedger()
        ledger.reserve("veh-1", Window(D(10), D(11)), "FA:load", expected_generation=0)
        with self.assertRaises(ConflictError):
            ledger.bulk_reserve(
                [
                    ("veh-2", Window(D(10), D(11)), "FB:load"),
                    ("veh-1", Window(D(10, 30), D(11, 30)), "FB:unload"),
                ],
                expected_generation=1,
            )
        self.assertEqual(ledger.reservations("veh-2"), ())
        self.assertEqual(len(ledger.reservations("veh-1")), 1)

    def test_task_cannot_start_before_predecessors_complete(self):
        planner = TurnaroundPlanner({"belt": ("veh-1",)})
        position = loading_task("position", 10, 0)
        load = loading_task("load", 20, 10, predecessors=("position",))
        planner.commit(planner.plan("FA", [position, load], D(10), D(12)))

        with self.assertRaises(RuntimeError):
            planner.transition("FA", "load", ServiceState.READY)
        planner.transition("FA", "position", ServiceState.READY)
        planner.transition("FA", "position", ServiceState.ACTIVE)
        with self.assertRaises(RuntimeError):
            planner.transition("FA", "load", ServiceState.ACTIVE)
        planner.transition("FA", "position", ServiceState.COMPLETE)
        planner.transition("FA", "load", ServiceState.READY)
        run = planner.transition("FA", "load", ServiceState.ACTIVE)
        self.assertEqual(run.state, ServiceState.ACTIVE)

    def test_release_returns_whole_order_without_residue(self):
        planner = TurnaroundPlanner({"belt": ("veh-1",)})
        planner.commit(planner.plan("FA", [loading_task()], D(10), D(12)))

        decision = planner.release("FA")
        self.assertTrue(decision.accepted)
        self.assertEqual(planner.ledger.reservations("veh-1"), ())
        self.assertNotIn("FA", planner.plans)

        again = planner.plan("FB", [loading_task()], D(10), D(12))
        self.assertEqual(again.runs["load"].planned, Window(D(10), D(10, 30)))
        self.assertTrue(planner.commit(again).accepted)

    def test_duplicate_commit_rejected_without_double_reservation(self):
        planner = TurnaroundPlanner({"belt": ("veh-1",)})
        plan = planner.plan("FA", [loading_task()], D(10), D(12))
        self.assertTrue(planner.commit(plan).accepted)
        self.assertFalse(planner.commit(plan).accepted)
        self.assertEqual(len(planner.ledger.reservations("veh-1")), 1)

    def test_workflow_race_has_deterministic_loser_and_no_residue(self):
        planner = TurnaroundPlanner({"belt": ("veh-1",)})
        control = RampWorkflowControl()
        plan_a = planner.plan("FA", [loading_task()], D(10), D(12))
        plan_b = planner.plan("FB", [loading_task()], D(10), D(12))

        receipt_a = control.commit_turnaround_dispatch("op-FA", "FA", D(10), lambda: planner.commit(plan_a))
        self.assertTrue(receipt_a.result.accepted)

        loser = planner.commit(plan_b)
        self.assertFalse(loser.accepted)
        with self.assertRaises(WorkflowRejected):
            control.commit_turnaround_dispatch(
                "op-FB", "FB", D(10), lambda: planner.commit(plan_b), requirements=(loser,)
            )

        self.assertEqual(control.stream.head, 1)
        held = planner.ledger.reservations("veh-1")
        self.assertEqual(len(held), 1)
        self.assertTrue(all(r.payload.startswith("FA:") for r in held))


if __name__ == "__main__":
    unittest.main()
