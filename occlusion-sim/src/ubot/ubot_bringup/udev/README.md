# Robot udev rules

These exist so `/dev/esp32` and `/dev/lidar` are stable names. Without them the
launch files point at whichever `/dev/ttyUSB*` the kernel happened to assign, which
changes between boots and between which device was plugged in first.

They were captured off the robot (`horizon`), where they are the only copy.

## Install on a new robot

```bash
sudo cp 99-horizon-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/esp32 /dev/lidar          # both should resolve
```

`99-horizon-serial.rules` maps:

| device | match | symlink |
|---|---|---|
| ESP32 motor bridge | 1a86:7523 | `/dev/esp32` |
| Oradar MS200 lidar | 1a86:55d4, serial 5890010080 | `/dev/lidar` |

The lidar rule matches on **serial number**, so a replacement unit needs its own
serial substituted (`udevadm info -a -n /dev/ttyUSB0 | grep serial`).

## OAK-D Lite

The depth camera needs the Luxonis rule as well, or depthai cannot claim the device
after it boots and re-enumerates:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
