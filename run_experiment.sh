dataset=nc-binding
model=svm
rep=bilnlm

if [[ ${1} == "download" ]]; then
    python BILN-LM/code/download_data.py downstream_data --collection downstream
fi
for dataset in c-binding c-cpp c-sol nc-binding nc-cpp; do
    for rep in ecfp molformer bilnlm pepclm pepland; do
        python BILN-LM/code/represent_peptides.py $dataset $rep
        for model in svm lightgbm; do
            for fp in mapc ecfp; do
                if [[ $fp == "mapc" ]]; then
                    sim=jaccard
                else
                    sim=tanimoto
                fi
                    python BILN-LM/code/evaluation.py $dataset $model $sim $fp $rep
                done
            done
        done
    done
done