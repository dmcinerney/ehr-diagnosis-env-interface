source activate /work/frink/mcinerney.de/envs/ehrenvint
PORT=8501
ssh login-01 -f -N -T -R ${PORT}:localhost:${PORT}
streamlit run model_explorer.py --server.port ${PORT}
