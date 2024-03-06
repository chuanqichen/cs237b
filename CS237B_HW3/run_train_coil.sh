
#python train_coil.py --scenario intersection --epochs 1000 --lr 5e-3
#python train_coil.py --scenario intersection --epochs 1000 --lr 1e-5 --restore

python test_coil.py --scenario intersection --goal left

python test_coil.py --scenario intersection --goal right

python test_coil.py --scenario intersection --goal straight

