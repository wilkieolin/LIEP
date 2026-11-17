# LIEP reproduction. Inside the image the working directory is the repo; on a host, see README (Setup without Docker).
IMAGE ?= liep
# run as the calling user, so files written into the checkout stay theirs
DOCKER_RUN = docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp -v $(CURDIR):/opt/liep -w /opt/liep

.PHONY: figures smoke image sif docker-figures docker-smoke

figures:            ## tier 1: summaries -> figure data -> Figs 6, 7 (CPU, minutes) and a comparison with the shipped tables
	scripts/figures.sh

smoke:              ## tier 2: one cheap run per figure family (fig6a grid; one Fig 7 cell on GPU; the LR controls on CPU)
	scripts/run_jobs.sh jobs/fig6a.txt
	scripts/run_jobs.sh -P 2 -g "run.jl lr " jobs/fig6c.txt
	scripts/run_jobs.sh -g "gaussian 7 1$$" jobs/fig7_2_damage.txt

image:              ## build the Docker image (needs the submodule: git submodule update --init)
	docker build -f docker/Dockerfile -t $(IMAGE) .

sif: image          ## Apptainer image for HPC hosts
	docker save $(IMAGE) -o liep.tar && apptainer build liep.sif docker/apptainer.def

docker-figures:
	$(DOCKER_RUN) $(IMAGE) make figures

docker-smoke:
	$(DOCKER_RUN) --gpus all $(IMAGE) make smoke
