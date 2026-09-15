# CAD source for the Inspection Robot body

`inspection_robot_01.f3z` is the Fusion 360 archive every mesh in
`../meshes/inspection/` came from. It lives here so the model and the URDF cannot
drift apart, and so a new machine needs one clone rather than a clone plus a file
someone has to remember to send.

Open it with **File → Open** in Fusion (it unpacks into your own project).

## Regenerating the meshes

The exporter is [fusion2urdf-ros2]. It will not read an arbitrary design: it turns
**top-level components** into links and nothing else, so the model has to be shaped
for it first.

| Rule | Why |
|---|---|
| Every link is a top-level component | Nested components are not links |
| Only a component's **own bodies** are meshed | Bodies in sub-components vanish from the STL but still count toward the mass — a silent error |
| The chassis component is named exactly `base_link` | The exporter looks for that name |
| Every other link needs a joint to `base_link`, with `base_link` as **Component 2** | Component 2 is the parent |
| Revolute joints with **no limits set** | Limits present → `revolute`; absent → `continuous`, which is what wheels need. Exactly one limit set aborts the export |

The exporter **edits the design while it runs** — it duplicates components and
renames the originals to `old_component`, and does not clean up. So: save, export,
then close **without saving**.

## Two traps that cost real time

**Copy/paste features stay bound to their source.** Copying bodies into new
components creates features that reference the originals; deleting those originals
later sends the copies to the world origin. Rebuild each copy as an independent
**base feature** instead.

**The exporter shifts every joint.** The ROS 1 exporter wrote all joint origins and
the base centre of mass shifted by a constant **(−16.4, +3.4, +19.7) mm** while
writing the meshes in the right place. This is nearly invisible: each mesh is drawn
relative to its own wrong joint, so the robot *looks* correct in RViz while every
wheel spins about an axle 26 mm off-centre and every sensor frame is 26 mm out.
The offset was recovered from the wheel mesh centres and subtracted. **Check the
joint origins against the mesh geometry after any re-export.**

`exports/fusion2urdf_2026-09-11/` is that raw export, kept for provenance, minus its
meshes — they were byte-identical to `../meshes/inspection/*.stl` and are not worth
storing twice. **Its joint origins carry the offset above; do not copy values out of
it.** The corrected geometry is in `../urdf/`.

## What the model is

Origin at the centre of the four wheels, at the chassis underside; X forward, Z up.
Wheelbase 255 mm, track 226.4 mm, wheel radius 32.5 mm, body 357 × 252 × 379 mm.
Masses come from CAD materials (PLA for printed parts, aluminium motors, rubber
tyres) and so exclude the battery, Pi and wiring.

Note the **effective** wheel separation used for odometry is 0.55 m, not the CAD
track — a 4-wheel skid-steer scrubs its tyres to rotate. See `CLAUDE.md` beside the
packages, and the comments in `ubot_bringup/config/ubot_controllers.yaml`.

The whole pipeline — CAD restructuring, the URDF package, WSL2 + ROS 2 Jazzy setup,
the Gazebo world, and the measurements behind these numbers — is written up in
`src/docs/hardware/Inspection_Robot_01_Setup_and_Simulation.pdf`.

[fusion2urdf-ros2]: https://github.com/dheena2k2/fusion2urdf-ros2
