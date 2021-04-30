.PHONY: app
app:
	docker-compose up

.PHONY: shell
shell:
	docker-compose run streamlit bash

.PHONY: clean
clean:
	git clean -fxn

.PHONY: build
build:
	(cd build && docker build -t covidsim-builder -f base.dockerfile ..)

.PHONY: extract
extract:
	docker create -ti --name dummy covidsim-builder bash
	docker cp dummy:/build/anaconda3 `pwd`/build/anaconda3
	docker rm -f dummy
