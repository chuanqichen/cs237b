echo "*** right ***"
echo "python train_ildist.py --scenario lanechange --goal right --epochs 1000 --lr 5e-3"
python train_ildist.py --scenario lanechange --goal right --epochs 1000 --lr 5e-3
#python train_ildist.py --scenario lanechange --goal right --epochs 28000 --lr 1e-5 --restore

echo "*** left ***"
echo "python train_ildist.py --scenario lanechange --goal left --epochs 1000 --lr 5e-3"
python train_ildist.py --scenario lanechange --goal left --epochs 1000 --lr 5e-3 
#python train_ildist.py --scenario lanechange --goal left --epochs 28000 --lr 1e-5 --restore
