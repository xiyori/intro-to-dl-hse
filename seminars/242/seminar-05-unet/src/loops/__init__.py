from .train import train, train_gan
from .train_supervised import train_supervised
from .test import test
from .predict import predict


def get_train_loop(name: str):
	loops = {
		"gan": train_gan,
		"supervised": train_supervised,
	}
	if name not in loops:
		raise ValueError(f"Unknown train loop '{name}'. Available loops: {', '.join(loops.keys())}")
	return loops[name]
