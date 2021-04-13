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
	docker-compose build
