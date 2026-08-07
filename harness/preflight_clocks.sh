#!/usr/bin/env bash
# Refuse to measure unless the board is in the condition declared in the paper:
# MAXN_SUPER with jetson_clocks applied (CPU and GPU pinned at max).
# jetson_clocks does not survive a reboot, so this is checked at every sweep.
preflight_clocks () {
  local g=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
  local gmin gmax cmin cmax mode
  gmin=$(cat "$g/min_freq" 2>/dev/null); gmax=$(cat "$g/max_freq" 2>/dev/null)
  cmin=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null)
  cmax=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)
  mode=$(nvpmodel -q 2>/dev/null | grep -i "power mode" -A0 | head -1)
  echo "preflight: nvpmodel='${mode}' GPU[min=${gmin} max=${gmax}] CPU[min=${cmin} max=${cmax}]"
  if [ "${gmin}" != "${gmax}" ] || [ "${cmin}" != "${cmax}" ]; then
    echo "PREFLIGHT FAIL: clocks are not pinned. Run:  sudo jetson_clocks" >&2
    return 1
  fi
  if [ "${gmax}" != "1020000000" ]; then
    echo "PREFLIGHT FAIL: GPU max ${gmax} != 1020000000 (expected MAXN_SUPER)" >&2
    return 1
  fi
  echo "preflight: OK (clocks pinned, MAXN_SUPER)"
  return 0
}
