FROM readthedocs/build:ubuntu-lts-latest

RUN apt-get update && apt-get install -y \
    r-base r-base-dev libtirpc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "install.packages(c('nnTensor'), repos='https://cloud.r-project.org')"

RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >> /etc/R/Rprofile.site
