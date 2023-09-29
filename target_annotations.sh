source activate /work/frink/mcinerney.de/envs/ehrenvint
PORT=8501
ssh login-01 -f -N -T -R ${PORT}:localhost:${PORT}
streamlit run target_annotations.py --server.port ${PORT}
#python -m pdb target_annotations.py
