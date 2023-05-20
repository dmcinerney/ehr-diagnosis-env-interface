export TRANSFORMERS_CACHE="/scratch/mcinerney.de/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/mcinerney.de/huggingface_cache"
module load anaconda3/3.7
module load cuda/11.8
source activate /work/frink/mcinerney.de/envs/ehrenvint
PORT=8503
ssh login-01 -f -N -T -R ${PORT}:localhost:${PORT}
streamlit run interface.py --server.port ${PORT}
