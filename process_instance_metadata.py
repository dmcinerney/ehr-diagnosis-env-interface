


if __name__ == '__main__':
    env = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=_instances,
        cache_path=args['data'][f'{split}_cache_path'],
        llm_name_or_interface=_llm_interface,
        fmm_name_or_interface=_fmm_interface,
        progress_bar=stqdm,
        **kwargs
    )
    if not annotate:
        postprocess_metadata = st.button('Postprocess Instance Metadata')
        if postprocess_metadata:
            def get_postprocessed_metadata(instance_metadata, env):
                new_rows = []
                for i, row in stqdm(instance_metadata.iterrows(), total=len(instance_metadata)):
                    new_rows.append(row.to_dict())
                    new_rows[-1].update(env.process_extracted_info({
                        k: v for k, v in new_rows[-1].items()
                        if k != 'episode_idx'}))
                return pd.DataFrame(new_rows)
            with st.spinner('Postprocessing Instance Metadata'):
                instance_metadata = get_postprocessed_metadata(
                    instance_metadata, env)
            import pdb; pdb.set_trace()
            # class_prevelances = instance_metadata
            # with open('class_prevelances.pkl', 'wb') as f:
            #     pkl.dump(class_prevelances, f)