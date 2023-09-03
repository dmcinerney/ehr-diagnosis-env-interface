from utils import get_args, get_dataset, get_environment, get_env_models
from tqdm import tqdm
import pandas as pd
import os


if __name__ == '__main__':
    args = get_args('config.yaml')
    df = get_dataset(args, args['process_metadata']['split'])
    llm_interface, fmm_interface = get_env_models(args)
    env = get_environment(
        args, args['process_metadata']['split'], df, llm_interface, fmm_interface)
    def get_postprocessed_metadata(instance_metadata, env):
        new_rows = []
        for i, row in tqdm(
                instance_metadata.iterrows(), total=len(instance_metadata)):
            new_rows.append(row.to_dict())
            new_rows[-1].update(env.process_extracted_info({
                k: v for k, v in new_rows[-1].items()
                if k != 'episode_idx'}))
        return pd.DataFrame(new_rows)
    instance_metadata = env.get_cached_instance_dataframe()
    instance_metadata = instance_metadata.sort_index().reset_index()
    instance_metadata = instance_metadata.rename(
        columns={'index': 'episode_idx'})
    instance_metadata = get_postprocessed_metadata(instance_metadata, env)
    if not os.path.exists(args['process_metadata']['outdir']):
        os.mkdir(args['process_metadata']['outdir'])
    instance_metadata.to_csv(
        os.path.join(
            args['process_metadata']['outdir'],
            args['process_metadata']['split'] + '.csv'),
        index=False)
