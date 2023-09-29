source activate /work/frink/mcinerney.de/envs/ehrenvint
PORT=8502
ssh login-01 -f -N -T -R ${PORT}:localhost:${PORT}
streamlit run interface.py --server.port ${PORT}
#python -m pdb interface.py
