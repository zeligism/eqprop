
EXPERIMENT_ID=${1:-"1"}

if [ "$EXPERIMENT_ID" == "1" ]; then
	python train.py \
	       --num-epochs 25 \
	       --free-iters 20 \
	       --clamped-iters 4 \
	       --beta 0.5 \
	       --dt 0.5 \
	       --layer-sizes 784 500 10 \
	       --learning-rates 0.1 0.05

elif [ "$EXPERIMENT_ID" == "2" ]; then
	python train.py \
	       --num-epochs 60 \
	       --free-iters 100 \
	       --clamped-iters 6 \
	       --beta 1.0 \
	       --dt 0.5 \
	       --layer-sizes 784 500 500 10 \
	       --learning-rates 0.4 0.1 0.01

elif [ "$EXPERIMENT_ID" == "3" ]; then
	python train.py \
	       --num-epochs 160 \
	       --free-iters 500 \
	       --clamped-iters 8 \
	       --beta 1.0 \
	       --dt 0.5 \
	       --layer-sizes 784 500 500 500 10 \
	       --learning-rates 0.128 0.032 0.008 0.002

elif [ "$EXPERIMENT_ID" == "nograd" ]; then
	python train.py --no-grad

elif [ "$EXPERIMENT_ID" == "load-and-test" ]; then
	python train.py -n 0 --load-model "models/model@epochs=2,iters=1971.pt"

elif [ "$EXPERIMENT_ID" == "graph" ]; then
	python train.py --graph --report-interval 1

else
	echo "Unrecognized argument: $EXPERIMENT_ID"
	echo "Running default training routine."
	python train.py
fi
