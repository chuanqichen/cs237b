echo "python train_il.py --scenario intersection --goal straight --epochs 1000 --lr 5e-3"
python train_il.py --scenario intersection --goal straight --epochs 1000 --lr 5e-3

echo "python train_il.py --scenario intersection --goal right --epochs 1000 --lr 5e-3"
python train_il.py --scenario intersection --goal right --epochs 1000 --lr 5e-3

echo "python train_il.py --scenario intersection --goal left --epochs 1000 --lr 5e-3"
python train_il.py --scenario intersection --goal left --epochs 1000 --lr 5e-3

#echo "python train_il.py --scenario intersection --goal all --epochs 1000 --lr 1e-4"
#python train_il.py --scenario intersection --goal all --epochs 1000 --lr 5e-3