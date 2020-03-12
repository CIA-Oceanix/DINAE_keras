#!/bin/sh

for type_obs in 'mod' 'obs' ; do
  python launch.py swot 0 ${type_obs}
  echo "NN-Learning with SWOT data ("${type_obs}")... Done"
  #for lag in $(seq 0 5); do
  for lag in '0' '5' ; do
      python launch.py nadir ${lag} ${type_obs}
      echo "NN-Learning with NADIR data ("${type_obs}") and lag "${lag}"... Done"
      python launch.py nadirswot ${lag} ${type_obs}
      echo "NN-Learning with NADIR/SWOT data ("${type_obs}") and lag "${lag}"... Done"
  done
done

