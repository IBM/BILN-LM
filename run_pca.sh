model=rf
sim=jaccard
fp=mapc

if [[ ${1} == "download" ]]; then
    python BILN-LM/code/download_data.py downstream_data --collection downstream
fi
for dataset in c-binding c-cpp c-sol nc-binding nc-cpp; do
    for rep in ecfp molformer bilnlm pepclm pepland; do
        python BILN-LM/code/represent_peptides.py $dataset $rep
    done

    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-molformer --pca
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-molformer
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-bilnlm --pca
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-bilnlm
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-pepland --pca
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-pepland
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-pepclm --pca
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp ecfp-pepclm
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp molformer-bilnlm --pca
    python BILN-LM/code/evaluation_mix.py $dataset $model $sim $fp molformer-bilnlm
    done
done