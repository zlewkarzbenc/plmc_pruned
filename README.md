# plmc_pruned
This repository is based on code originally published under an MIT Licenses by dpxiong (https://github.com/dpxiong/PLMC) and raghvendra5688 (https://github.com/raghvendra5688/BCrystal). This version is provided as-is with **no endorsement of the original implementations or associated claims**.

## PLMC modification
PLMC is a deep learning framework for protein crystallization prediction with protein language embeddings and handcrafted features. This code provides a version of PLMC that doesn't use protein language embeddings (for various reasons).

## Installation
  - R requirements
    - Run R REPL by running the following: `R`
    -  Install R libraries
       1.  Interpol (do `install.packages("https://cran.r-project.org/src/contrib/Archive/Interpol/Interpol_1.3.1.tar.gz", repos = NULL, type = "source")` )
       2.  bio3d    (do `install.packages('bio3d')` )
       3.  doParallel (do `install.packages('doParallel')`)
       4.  zoo      (do `install.packages('zoo')`)
       
    Quit R REPL: `quit()` 
 
  - SCRATCH (version SCRATCH-1D release 1.2) (http://scratch.proteomics.ics.uci.edu, Downloads: http://download.igb.uci.edu/#sspro)
    1. Run `wget http://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz`
    2. Run `tar -xvzf SCRATCH-1D_1.2.tar.gz`
    3. Run `cd SCRATCH-1D_1.2`
    4. Run `perl install.pl`
    5. Run `cd ..`
    6. Replace the blast in `SCRATCH-1D_1.2/pkg/blast-2.2.26` with a 64 bit version of `blast-2.2.26` if you are running on a 64 bit machine (`ftp://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/`).
    
  - DISOPRED (version 3.16) (http://bioinfadmin.cs.ucl.ac.uk/downloads/DISOPRED/)
    1. Run `wget http://bioinfadmin.cs.ucl.ac.uk/downloads/DISOPRED/DISOPRED3.16.tar.gz`
    2. Run `tar -xvzf DISOPRED3.16.tar.gz`
    3. Run `cd DISOPRED/src/`
    4. Run `make clean; make; make install`
    5. In `run_disopred.pl` file within the DISOPRED folder put `my $NCBI_DIR = <path-to-SCRATCH-folder>/pkg/blast-2.2.26/bin`
    6. In `run_disopred.pl` file also put `my $SEQ_DB = <path-to-SCRATCH-folder>/pkg/PROFILpro_1.2/data/uniref50/uniref50`.

### Creating a conda environment
After cloning the repository, please run the following lines to install an environment for PLMC
```
conda create -n plmc_pruned python=3.9
conda activate plmc_pruned
conda install numpy=1.23.5
conda install -c pytorch pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
conda install -c conda-forge scikit-learn=1.0.2
```

### Data preparation
Download the data at this [address](https://zenodo.org/record/6475529/), and uncompress it to the current directory.

### Training PLMC
Execute the following commands:
```
python main.py --rawpath ./data/CRYS_DS
```

### Predicting crystallization outcome of your own protein:
Execute in the command line
  1. `Rscript --vanilla features_PaRSnIP_v2.R yourfile.fasta`
  2. 
