r"""Collisions: raw Gazebo contacts in, one event per impact out.

    python3 -m ubot_eval.contact --selftest        # debounce logic, no ROS

--------------------------------------------------------------------------
WHY DEBOUNCE
--------------------------------------------------------------------------
The Contact system republishes while surfaces stay touching, at roughly the
physics rate. One wall impact during Part 0 produced **3033 messages**. Counted
raw, a single crash outranks thirty separate ones, and "collisions per metre"
becomes a measure of how long the robot leaned on the wall afterwards.

An impact is therefore a *gap-separated run* of contact messages: contacts that
stop for `gap_s` and then resume are two events. 0.5 s is long enough to bridge
the physics chatter of one collision and short enough to separate a bounce from
a fresh hit at 0.15 m/s (7.5 cm of travel).

--------------------------------------------------------------------------
WHAT COUNTS AS A COLLISION HERE
--------------------------------------------------------------------------
Chassis only. The sensor is on `base_link`, whose collisions are the body and
mast boxes; the wheels are the only part that touches the floor in normal
driving, so every reported contact is already a real collision and there is no
ground-contact filter to get subtly wrong (18 mm of chassis clearance).

Two silent traps, both already sprung once and documented in
ubot_gazebo.urdf.xacro:

* fixed-joint lumping renames the collisions to
  `base_footprint_fixed_joint_lump__base_link_collision*`, so a sensor naming
  the declared collisions matches nothing and reports zero collisions for ever -
  which reads exactly like a robot that never crashes;
* the Contact system ignores the sensor's `<topic>` and publishes on the scoped
  world path, so bridging a short name yields a ROS topic that exists, is
  subscribed, and never receives a message.

Both mean the *absence* of contacts is not evidence. `ContactMonitor.health()`
exists so an episode can record whether the stream was ever alive, rather than
silently scoring a broken sensor as a clean run.
"""
from __future__ import annotations

import argparse
import sys

#: Contacts separated by more than this are separate impacts.
GAP_S = 0.5


def debounce(times, gap_s: float = GAP_S):
    """Start time of each impact, from the timestamps of contact messages."""
    events = []
    last = None
    for t in sorted(times):
        if last is None or t - last > gap_s:
            events.append(t)
        last = t
    return events


def summarize(times, names=(), path_length_m: float | None = None,
              gap_s: float = GAP_S) -> dict:
    """The `collisions` field of an episode summary.

    `per_metre` is None rather than infinity when nothing was travelled: a robot
    that collided without moving has an undefined rate, and a pooled statistic
    must not inherit an infinity from it. The plan pools per-metre rates across
    episodes by summing numerators and denominators, which needs the raw count
    and the distance, both of which are kept here.
    """
    events = debounce(times, gap_s)
    per_metre = None
    if path_length_m and path_length_m > 0:
        per_metre = len(events) / float(path_length_m)
    return dict(count=len(events),
                events=[round(float(t), 3) for t in events],
                first_s=round(float(events[0]), 3) if events else None,
                raw_messages=len(list(times)),
                names=sorted(set(names))[:8],
                path_length_m=path_length_m,
                per_metre=per_metre)


class ContactMonitor:
    """Subscribes `/bumper/contacts` and remembers when the chassis was touched.

    ROS imports are lazy so the debounce logic above can be tested anywhere.
    Timestamps come from the message header, i.e. SIM time, so they line up with
    the trajectory and the bag rather than with the wall clock.
    """

    def __init__(self, node, *, topic: str = "/bumper/contacts", gap_s: float = GAP_S):
        from ros_gz_interfaces.msg import Contacts
        self.node = node
        self.topic = topic
        self.gap_s = gap_s
        self.times: list[float] = []
        self.names: list[str] = []
        self.messages = 0
        node.create_subscription(Contacts, topic, self._on_contacts, 50)

    def _on_contacts(self, msg) -> None:
        self.messages += 1
        # Gazebo publishes empty Contacts frequently; an empty list is the
        # sensor saying "nothing is touching", not a collision.
        if not msg.contacts:
            return
        stamp = msg.header.stamp
        self.times.append(stamp.sec + stamp.nanosec * 1e-9)
        for c in msg.contacts[:4]:
            self.names.append(f"{c.collision1.name} | {c.collision2.name}")

    @property
    def collided(self) -> bool:
        return bool(self.times)

    def health(self) -> dict:
        """Whether the stream was alive at all. Absence of contacts is not proof."""
        return dict(topic=self.topic, messages_seen=self.messages,
                    stream_alive=self.messages > 0)

    def summary(self, path_length_m: float | None = None) -> dict:
        d = summarize(self.times, self.names, path_length_m, self.gap_s)
        d.update(self.health())
        return d

    def reset(self) -> None:
        self.times.clear()
        self.names.clear()
        self.messages = 0


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.contact (debounce logic, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<52} {detail}")

    # The Part 0 measurement: one wall hit, 3033 messages at ~50 Hz.
    one_hit = [i / 50.0 for i in range(3033)]
    ev = debounce(one_hit)
    check("3033 raw messages from one wall hit -> 1 event", len(ev) == 1,
          f"{len(ev)} event(s) from {len(one_hit)} messages")

    # two impacts separated by a clear gap
    two = [i / 50.0 for i in range(100)] + [10.0 + i / 50.0 for i in range(100)]
    check("two impacts 8 s apart -> 2 events", len(debounce(two)) == 2)

    # a bounce inside one impact is not a second collision
    bounce = [i / 50.0 for i in range(50)] + [1.2 + i / 50.0 for i in range(50)]
    check("0.2 s bounce stays one event", len(debounce(bounce)) == 1,
          f"{len(debounce(bounce))} event(s)")
    check("a 0.6 s gap does separate them", len(debounce(
        [i / 50.0 for i in range(50)] + [1.6 + i / 50.0 for i in range(50)])) == 2)

    check("no contacts -> no events", debounce([]) == [])

    # first_s is the first impact, and it survives being fed out of order
    s = summarize([5.0, 0.4, 5.02, 0.42], path_length_m=10.0)
    check("first_s is the first impact, order-independent", s["first_s"] == 0.4,
          f"first_s={s['first_s']}, count={s['count']}")
    check("per metre uses events, not raw messages",
          abs(s["per_metre"] - 0.2) < 1e-9, f"{s['per_metre']:.3f} /m")

    s0 = summarize([1.0], path_length_m=0.0)
    check("collision with no travel -> per_metre None, not inf",
          s0["per_metre"] is None and s0["count"] == 1)
    s_none = summarize([1.0], path_length_m=None)
    check("unknown distance -> per_metre None", s_none["per_metre"] is None)

    clean = summarize([], path_length_m=12.0)
    check("clean run reports 0 and a real denominator",
          clean["count"] == 0 and clean["per_metre"] == 0.0 and clean["path_length_m"] == 12.0)

    s_names = summarize([1.0], names=["hull | wall", "hull | wall", "hull | pillar"],
                        path_length_m=3.0)
    check("collision provenance kept, de-duplicated", s_names["names"] ==
          ["hull | pillar", "hull | wall"], str(s_names["names"]))

    check("raw message count retained for the record",
          summarize(one_hit)["raw_messages"] == 3033)

    # the contract with schema: a collided episode must be scored `collided`
    from ubot_eval.schema import classify, validate
    outcome, _ = classify(readiness_ok=True, collided=bool(debounce(one_hit)),
                          nav2_status="SUCCEEDED", gt_distance_to_goal_m=0.05,
                          timed_out=False)
    check("collision outranks nav2 SUCCEEDED", outcome == "collided", outcome)
    problems = validate(dict(schema_version=1, campaign="c", arm="a", world="w",
                             course="x", trial=1, outcome="reached",
                             shortest_path_m=3.0, path_length_m=3.2,
                             collisions=summarize(one_hit, path_length_m=3.2)))
    check("schema rejects collisions filed under a clean outcome", bool(problems))

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else ap.print_help())


if __name__ == "__main__":
    main()
