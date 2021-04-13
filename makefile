.PHONY: app
app:
	docker-compose up

.PHONY: shell
shell:
	docker-compose run streamlit bash

.PHONY: build
build:
	docker-compose build
