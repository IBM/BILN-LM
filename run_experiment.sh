dataset=nc-binding
model=svm
rep=bilnlm

python BILN-LM-OSS/code/represent_peptides.py $dataset $rep

for fp in map4 maccs; do
    for sim in tanimoto dice cosine rogot-goldberg sokal; do
        python BILN-LM-OSS/code/evaluation.py $dataset $model $sim $fp $rep
    done
done