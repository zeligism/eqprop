
RUN_TYPE=${1:-"1"}

if [ "$RUN_TYPE" == "1" ]; then
	python train.py \
	       --num-epochs 25 \
	       --free-iters 20 \
	       --clamped-iters 4 \
	       --beta 0.5 \
	       --dt 0.5 \
	       --layer-sizes 784 500 10 \
	       --learning-rates 0.1 0.05

elif [ "$RUN_TYPE" == "2" ]; then
	python train.py \
	       --num-epochs 60 \
	       --free-iters 100 \
	       --clamped-iters 6 \
	       --beta 1.0 \
	       --dt 0.5 \
	       --layer-sizes 784 500 500 10 \
	       --learning-rates 0.4 0.1 0.01

elif [ "$RUN_TYPE" == "3" ]; then
	python train.py \
	       --num-epochs 160 \
	       --free-iters 500 \
	       --clamped-iters 8 \
	       --beta 1.0 \
	       --dt 0.5 \
	       --layer-sizes 784 500 500 500 10 \
	       --learning-rates 0.128 0.032 0.008 0.002

elif [ "$RUN_TYPE" == "nograd" ]; then
	python train.py --no-grad

elif [ "$RUN_TYPE" == "load-and-test" ]; then
	python train.py -n 0 --load-model "models/model@epochs=2,iters=1971.pt"

elif [ "$RUN_TYPE" == "graph" ]; then
	python train.py --graph --report-interval 1

elif [ "$RUN_TYPE" == "spiking" ]; then
	python train.py --spiking

elif [ "$RUN_TYPE" == "spiking-nograd" ]; then
	python train.py --spiking --no-grad

elif [ "$RUN_TYPE" == "continual" ]; then
	python train.py --continual --clamped-iters 5 --learning-rates 0.02 0.01

elif [ "$RUN_TYPE" == "continual-nograd" ]; then
	python train.py --continual --clamped-iters 5 --learning-rates 0.02 0.01 --no-grad

else
	echo "Unrecognized argument: $RUN_TYPE"
	echo "Running default training routine."
	python train.py
fi
