#!/bin/sh

python launch.py swot 0 mod
echo "NN-Learning with SWOT data... Done"
python launch.py nadir 0 mod
echo "NN-Learning with NADIR data and lag 0... Done"
python launch.py nadirswot 0 mod
echo "NN-Learning with NADIR/SWOT data and lag 0... Done"
python launch.py nadir 1 mod
echo "NN-Learning with NADIR data and lag 1... Done"
python launch.py nadirswot 1 mod
echo "NN-Learning with NADIR/SWOT data and lag 1... Done"
python launch.py nadir 2 mod
echo "NN-Learning with NADIR data and lag 2... Done"
python launch.py nadirswot 2 mod
echo "NN-Learning with NADIR/SWOT data and lag 2... Done"
python launch.py nadir 3 mod
echo "NN-Learning with NADIR data and lag 3... Done"
python launch.py nadirswot 3 mod
echo "NN-Learning with NADIR/SWOT data and lag 3... Done"
python launch.py nadir 4 mod
echo "NN-Learning with NADIR data and lag 4... Done"
python launch.py nadirswot 4 mod
echo "NN-Learning with NADIR/SWOT data and lag 4... Done"
python launch.py nadir 5 mod
echo "NN-Learning with NADIR data and lag 5... Done"
python launch.py nadirswot 5 mod
echo "NN-Learning with NADIR/SWOT data and lag 5... Done"
