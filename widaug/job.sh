
#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=profner
#SBATCH --mem-per-cpu=10000
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pcalleja@fi.upm.es
#SBATCH --output=out-%j.log
##------------------------ End job description ------------------------

module purge && module load CUDA


source /home/s730/s730251/projects/envprof/bin/activate

# rutas absolutas! 
srun python3 ner_training.py training_or.tsv valid_spacy.txt
srun python3 ner_training.py training_50.tsv valid_spacy.txt
srun python3 ner_training.py training_30.tsv valid_spacy.txt
srun python3 ner_training.py training_10.tsv valid_spacy.txt

deactivate



