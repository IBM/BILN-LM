dataset=nc-binding
model=svm
rep=bilnlm

for model in svm lightgbm; do
    for dataset in c-binding c-cpp c-sol nc-binding nc-cpp; do
        for rep in ecfp molformer bilnlm pepclm pepland; do
            python BILN-LM-OSS/code/represent_peptides.py $dataset $rep
            for fp in mapc ecfp; do
                if [[ $fp == "mapc" ]]; then
                    sim=jaccard
                else
                    sim=tanimoto
                fi
                    python BILN-LM-OSS/code/evaluation.py $dataset $model $sim $fp $rep
                done
            done
        done
    done
done