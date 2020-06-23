#!/bin/bash
debootstrap_dir=debootstrap
root_filesystem=debootstrap.ext2.qcow2
linux_image="$(printf "${debootstrap_dir}/boot/vmlinuz-"*)"
../../qemu-opt/x86_64-softmmu/qemu-system-x86_64 \
  -append 'console=ttyS0 root=/dev/sda' \
  -drive "file=${root_filesystem},format=qcow2" \
  -enable-kvm \
  -serial mon:stdio \
  -m 2G \
  -kernel "${linux_image}" \
  -device rtl8139,netdev=net0 \
  -netdev user,id=net0
