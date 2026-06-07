#!/usr/bin/env python3
"""
Bring your own robot — a custom RobotConnector.

Reflex ships connectors for a few arms (yam_bimanual and friends), but any
robot works: implement this small interface, point your YAML config at it, and
`reflex connect` drives it. No PR to the SDK needed — Reflex imports your class
by its `module:Class` path.

Wire it up in your connect config:

    hardware:
      kind: "custom_robot:XArm5Connector"   # module:Class — Reflex imports this
      config:                               # passed straight to __init__ as kwargs
        ip: 192.168.1.220
        hz: 25

The contract (reflex.connectors.base.RobotConnector):
    start()                            open drivers + cameras
    get_observation() -> Observation   one observation per control step
    apply_action(ActionChunk)          apply an action chunk to the hardware
    safe_stop(reason="")               halt motion safely (do implement this)
    stop()                             release hardware

    Observation(state: list[float], cameras: {name: image}, task: str)
    ActionChunk(actions: [[...], ...])   # shape [T, action_dim]

The example below is an xArm5 skeleton. The xArm-specific lines are commented
out so the file imports without the SDK installed — fill them in for your arm.
"""
from __future__ import annotations

from reflex.connectors.base import ActionChunk, Observation, RobotConnector


class XArm5Connector(RobotConnector):
    kind = "custom_robot"

    def __init__(self, ip: str = "192.168.1.220", hz: int = 25, **_: object) -> None:
        self.ip = ip
        self.hz = hz
        self._arm = None
        self._cams: dict[str, object] = {}

    def start(self) -> None:
        # Open the robot SDK + cameras. For an xArm5:
        #   from xarm.wrapper import XArmAPI
        #   self._arm = XArmAPI(self.ip)
        #   self._arm.motion_enable(True)
        #   self._arm.set_mode(1)      # servo / online-trajectory mode
        #   self._arm.set_state(0)
        #   import cv2
        #   self._cams = {"front": cv2.VideoCapture(0)}
        ...

    def get_observation(self) -> Observation:
        # Read proprioception + camera frames. `state` is the numeric vector the
        # model consumes — match the order/units your model was trained on.
        #   _, angles = self._arm.get_servo_angle(is_radian=True)   # 5 joints
        #   gripper = self._arm.get_gripper_position()[1]
        #   state = list(angles) + [gripper]
        #   frames = {name: cam.read()[1] for name, cam in self._cams.items()}
        state: list[float] = [0.0] * 6        # 5 xArm5 joints + gripper
        frames: dict[str, object] = {}        # {"front": np.ndarray HxWx3}
        return Observation(state=state, cameras=frames, task="pick up the block")

    def apply_action(self, action: ActionChunk) -> None:
        # action.actions is [T, action_dim]. Map the model's action dim onto the
        # arm's DOF. Closed-loop: apply ONE step, then take a fresh observation
        # (the runner calls get_observation() again next tick).
        for step in action.actions:
            #   self._arm.set_servo_angle_j(step[:5], is_radian=True, wait=False)
            #   self._arm.set_gripper_position(step[5])
            break

    def safe_stop(self, reason: str = "") -> None:
        # Bring the arm to a safe state on Ctrl-C / errors. IMPLEMENT THIS for
        # real hardware — it is what keeps the arm from latching mid-motion.
        #   self._arm.set_state(4)   # stop motion
        ...

    def stop(self) -> None:
        # Release drivers + cameras.
        #   if self._arm: self._arm.disconnect()
        #   for cam in self._cams.values(): cam.release()
        ...


if __name__ == "__main__":
    # Smoke test: confirm the class resolves through the Reflex registry the
    # same way `reflex connect` will load it from your config.
    from reflex.connectors import get_connector_class

    cls = get_connector_class("custom_robot:XArm5Connector")
    print(f"resolved: {cls.__name__} (kind={cls.kind!r})")
