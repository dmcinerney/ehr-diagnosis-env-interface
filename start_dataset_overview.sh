source activate /work/frink/mcinerney.de/envs/ehrenvint
PORT=8505
ssh login-01 -f -N -T -R ${PORT}:localhost:${PORT}
streamlit run dataset_overview.py --server.port ${PORT}
