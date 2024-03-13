.PHONY: all clear x_vectors k_means pca librispeech

all: x_vectors k_means pca

x_vectors:
	@if [ ! -f x_vectors.json ]; then \
		cp x_vectors.json.sample x_vectors.json; \
		echo "Copying x_vectors.json.sample -> x_vectors.json"; \
	fi
	@python3 x_vectors.py x_vectors.json

k_means:
	@if [ ! -f k_means.json ]; then \
		cp k_means.json.sample k_means.json; \
		echo "Copying k_means.json.sample -> k_means.json"; \
	fi
	@python3 k_means.py k_means.json

pca:
	@if [ ! -f pca.json ]; then \
		cp pca.json.sample pca.json; \
		echo "Copying pca.json.sample -> pca.json"; \
	fi
	@python3 pca.py pca.json

librispeech:
	@if [ ! -f librispeech.json ]; then \
		cp librispeech.json.sample librispeech.json; \
		echo "Copying librispeech.json.sample -> librispeech.json"; \
	fi
	@python3 x_vectors.py librispeech.json
	@python3 k_means.py librispeech.json
	@python3 pca.py librispeech.json

common_voice:
	@if [ ! -f common_voice.json ]; then \
		cp common_voice.json.sample common_voice.json; \
		echo "Copying common_voice.json.sample -> common_voice.json"; \
	fi
	@python3 x_vectors.py common_voice.json
	@python3 k_means.py common_voice.json
	@python3 pca.py common_voice.json

clear:
	@echo "Removing x_vectors.json"
	@rm -f x_vectors.json
	@echo "Removing k_means.json"
	@rm -f k_means.json
	@echo "Removing pca.json"
	@rm -f pca.json
	@echo "Removing librispeech.json"
	@rm -f librispeech.json
