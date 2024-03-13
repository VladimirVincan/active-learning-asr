.PHONY: all clear librispeech common_voice

all: librispeech common_voice

librispeech:
	@if [ ! -f librispeech.json ]; then \
		cp librispeech.json.sample librispeech.json; \
		echo "Copying librispeech.json.sample -> librispeech.json"; \
	fi
	@python3 DatasetDownloader.py librispeech.json

common_voice:
	@if [ ! -f common_voice.json ]; then \
		cp common_voice.json.sample common_voice.json; \
		echo "Copying common_voice.json.sample -> common_voice.json"; \
	fi
	@python3 DatasetDownloader.py common_voice.json

clear:
	@echo "Removing librispeech.json"
	@rm -f librispeech.json
	@echo "Removing common_voice.json"
	@rm -f common_voice.json
