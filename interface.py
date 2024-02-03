from gc import disable
from utils import *
import streamlit as st
import pandas as pd
import numpy as np
from ehr_diagnosis_agent.models.actor import InterpretableDirichletActor, \
    InterpretableNormalActor, InterpretableBetaActor, InterpretableDeltaActor
from ehr_diagnosis_agent.models.observation_embedder import \
    BertObservationEmbedder
import torch
from torch.distributions import Categorical, Bernoulli, kl_divergence
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep
import pickle as pkl
import random
from collections import Counter
from torch import nn


actor_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
    'delta': InterpretableDeltaActor,
}
recommended_reward_types = {
    'normal': ['continuous_dependent', 'ranking'],
    'dirichlet': ['continuous_dependent', 'ranking'],
    'beta': ['continuous_independent', 'ranking'],
    'delta': ['continuous_independent', 'continuous_dependent', 'ranking'],
}


@st.cache_data
def get_checkpoint_args(args, actor_checkpoint):
    split_ckpt_path = args['models'][actor_checkpoint].split('/')
    config_path = '/'.join(split_ckpt_path[:-1] + ['config.yaml'])
    return OmegaConf.load(config_path)


@st.cache_resource
def get_actor(args, actor_checkpoint):
    checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
    actor_params = checkpoint_args.actor.value['{}_params'.format(
        checkpoint_args.actor.value.type)]
    actor_params.update(checkpoint_args.actor.value['shared_params'])
    # static bias params is set differently during inference
    if args['static_bias_params'] is not None:
        actor_params['static_bias_params'] = args['static_bias_params']
    actor = actor_types[checkpoint_args.actor.value.type](actor_params)
    state_dict = torch.load(args['models'][actor_checkpoint])['actor']
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


@st.cache_data
def sample_action(args, observation, actor_checkpoint, sample_type='mean'):
    actor = get_actor(args, actor_checkpoint)
    with torch.no_grad():
        parameter_votes_info = actor.get_dist_parameter_votes(observation)
        parameters_info = actor.votes_to_parameters(parameter_votes_info)
    dist = actor.parameters_to_dist(*parameters_info['params'])
    # action = dist.sample()
    if sample_type == 'mean':
        if isinstance(actor, InterpretableDirichletActor) or isinstance(
                actor, InterpretableBetaActor):
            # action = torch.log(dist.base_dist.mode)
            action = torch.log(dist.base_dist.mean)
            action_stddev = dist.base_dist.stddev
        else:
            # action = dist.mode
            action = dist.mean
            action_stddev = dist.stddev
    else:
        raise Exception
    return parameter_votes_info, parameters_info, action.detach().numpy(), \
        action_stddev


@st.cache_resource
def get_instance_metadata_with_model_output_metadata(
        _env, outputs_to_add, args, split):
    df = get_instance_metadata(_env, args, split)
    if outputs_to_add != 'No model selected':
        model_outputs = pd.read_csv(
            args['model_outputs'][split][outputs_to_add])
        df = df.merge(model_outputs, how='left', on='episode_idx')
    return df


def make_plot(options, probs, not_dist=False, ylabel='probability', ax=None):
    data = pd.DataFrame({
        'option': options,
        ylabel: probs,
    })
    # st.bar_chart(data, x='option', y='probability')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        fig = None
    chart = sns.barplot(data, x='option', y=ylabel, ax=ax)
    if not not_dist:
        chart.set_ylim(0, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 20)
    for i, v in enumerate(probs):
        ax.text(i - .2, v.item() + 0.01, '{:.1f}%'.format(v.item() * 100))
    if fig is not None:
        st.pyplot(fig)


def get_annotations(ann_path):
    if not os.path.exists(ann_path):
        return pd.DataFrame([])
    df = {}
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            with open(os.path.join(ann_path, filename), 'rb') as f:
                data = pkl.load(f)
                value = next(iter(data.values()))
                if 'model_anns' in value.keys():
                    if isinstance(value['model_anns'], dict):
                        value.update({
                            (k1, k2): v2
                            for k1, v1 in value['model_anns'].items()
                            for k2, v2 in v1.items()})
                    del value['model_anns']
                df.update(data)
    return pd.DataFrame.from_dict(df, orient='index')


def get_complementary_annotations(args, split):
    complementary_annotations_dirs = [
        args['annotations']['path']] + args['annotations']['complementary']
    comp_anns = pd.DataFrame([])
    for comp_ann_path in complementary_annotations_dirs:
        if os.path.exists(os.path.join(comp_ann_path, split)):
            for annotator_subdir in os.listdir(
                    os.path.join(comp_ann_path, split)):
                new_comp_anns = get_annotations(os.path.join(
                    comp_ann_path, split, annotator_subdir))
                new_comp_anns['annotator'] = [
                    annotator_subdir] * len(new_comp_anns)
                new_comp_anns['base_path'] = [
                    comp_ann_path] * len(new_comp_anns)
                comp_anns = pd.concat([comp_anns, new_comp_anns])
    return comp_anns


# class ReloadInstances:
#     def __init__(self, split):
#         self.split = split
#     def __call__(self):
#         del st.session_state[f'balanced_instance_sample_{self.split}']


st.set_page_config(layout="wide")
st.title('EHR Diagnosis Environment Visualizer')
args = get_args('config.yaml')
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = get_splits(args)
with st.sidebar:
    annotate = st.checkbox('Annotate', value=True)
    if annotate:
        if st.button('Restart Report Timer'):
            if 'episode' in st.session_state and \
                    'display_report_timer' in \
                        st.session_state['episode'].keys():
                del st.session_state['episode']['display_report_timer']
        num_evidence_to_annotate = st.number_input(
            'Number of evidence snippets to annotate', min_value=1, value=10)
    num_evidence_at_a_time = st.number_input(
        'Amount of evidence to reveal at a time.', min_value=1, value=1)
    show_remaining_anns = st.checkbox('Show remaining evidence')
    show_state = st.checkbox('Show state')
    control_state = False
    pick_how_to_sort_evidence = False
    show_evidence = False
    if not annotate:
        control_state = st.checkbox('Control state')
        pick_how_to_sort_evidence = st.checkbox('Pick how to sort evidence')
        show_evidence = st.checkbox('Show evidence')
    show_instance_metadata = st.checkbox('Show instance metadata')
    ignore_selected_evidence = st.checkbox('Ignore selected evidence types')
    if annotate:
        use_actor = True
    else:
        use_actor = st.checkbox('Use actor models', value=True)
    st.write('#### Environment')
    use_random_start_idx = st.checkbox(
        'Start at a random timestep', value=True,
        key='start at random timestep',
        on_change=reset_episode_state)
    reward_type = args['env']['reward_type']
    st.write(f'Reward type: {reward_type}')
    split = st.selectbox(
        'Dataset Split', splits,
        index=splits.index('val1') if 'val1' in splits else 0,
        key='split',
        on_change=reset_episode_state)
df = get_dataset(args, split)
llm_interface, fmm_interface = get_env_models(args)
env = get_environment(
    args, split, df, llm_interface, fmm_interface)
with st.sidebar:
    st.write('#### Instances')
    if st.checkbox('Require Metadata', value=True):
        if split in args['model_outputs'].keys():
            options = ['No model selected'] + list(
                args['model_outputs'][split].keys())
            outputs_to_add = st.selectbox(
                'Add pre-computed model outputs to the metadata by selecting a model.',
                options)
        else:
            outputs_to_add = 'No model selected'
        instance_metadata = get_instance_metadata_with_model_output_metadata(
            env, outputs_to_add, args, split)
        filtered_instance_metadata = instance_metadata
        if 'is valid timestep' in filtered_instance_metadata.columns:
            min_num_reports = st.number_input(
                'Minimum Number of Reports', min_value=0, value=10,
                # on_change=ReloadInstances(split),
                )
        else:
            min_num_reports = 0
        show_cached_evidence = st.checkbox(
            'Only show instances with cached evidence', value=True,
            disabled=min_num_reports > 0,
            # on_change=ReloadInstances(split),
            key='disabled' if min_num_reports > 0 else 'not_disabled')
        if (min_num_reports > 0 or show_cached_evidence) and \
                'is valid timestep' in filtered_instance_metadata.columns:
            filtered_instance_metadata = filtered_instance_metadata[
                filtered_instance_metadata['is valid timestep'].apply(
                    lambda x: x == x and sum(x) >= min_num_reports)]
        comp_anns = None
        if annotate:
            comp_anns = get_complementary_annotations(args, split)
            if len(comp_anns) > 0:
                comp_anns = comp_anns[comp_anns.num_reports >= min_num_reports]
                partially_annotated_instances = set(comp_anns.instance)
                max_annotated_instance = max(
                    [int(x.split()[2]) for x in partially_annotated_instances])
                partially_annotated_instance_metadata = filtered_instance_metadata[
                    filtered_instance_metadata['episode_idx'].apply(
                        lambda x: x + 1 <= max_annotated_instance)]
            else:
                max_annotated_instance = None
                partially_annotated_instance_metadata = None
            if f'balanced_instance_sample_{split}' not in st.session_state.keys() \
                    or st.button('Re-sample Instances'):
                if max_annotated_instance is not None:
                    filtered_instance_metadata = filtered_instance_metadata[
                        filtered_instance_metadata['episode_idx'].apply(
                            lambda x: x + 1 > max_annotated_instance)]
                positives = filtered_instance_metadata[
                    filtered_instance_metadata['target diagnosis countdown'].apply(
                        lambda x: x == x and len(x[0]) > 0)]
                # TODO add in parameter to control sampling
                negatives = filtered_instance_metadata[
                    filtered_instance_metadata['target diagnosis countdown'].apply(
                        lambda x: x == x and len(x[0]) == 0)].sample(
                            n=len(positives))
                balanced_instance_sample = pd.concat([positives, negatives])
                if partially_annotated_instance_metadata is not None:
                    balanced_instance_sample = pd.concat([
                        partially_annotated_instance_metadata,
                        balanced_instance_sample])
                st.session_state[f'balanced_instance_sample_{split}'] = \
                    balanced_instance_sample
            filtered_instance_metadata = st.session_state[
                f'balanced_instance_sample_{split}']
        filter_instances_string = st.text_input(
            'Type a lambda expression in python that filters instances using their'
            ' cached metadata.')
        if filter_instances_string != '':
            filtered_instance_metadata = filtered_instance_metadata[
                filtered_instance_metadata.apply(
                    eval(filter_instances_string), axis=1)]
        valid_instances = filtered_instance_metadata.sort_values(
            'episode_idx').episode_idx
    else:
        valid_instances = np.array(range(len(df)))
    num_valid = len(valid_instances)
    if num_valid == 0:
        st.warning('No results after filtering.')
        st.stop()
    st.divider()
    st.write('Configuration')
    st.write(args)
    st.divider()
    spinner_container = st.container()
if show_instance_metadata:
    with st.expander('Instance metadata'):
        st.write(filtered_instance_metadata[:100])
        st.write('...')
        if 'all_targets' in filtered_instance_metadata.columns:
            counts = Counter()
            rows = filtered_instance_metadata[
                ~filtered_instance_metadata.all_targets.isna()]
            for i, row in rows.iterrows():
                for x in eval(row.all_targets):
                    counts[x] += 1
            # st.write(counts)
            options = sorted(list(counts.keys()))
            probs = np.array([counts[o] for o in options]) / len(rows)
            with st.columns([2, 5])[0]:
                make_plot(options, probs)


if annotate:
    annotator_name = st.text_input('Annotator Name')
instance_picking_warning = st.empty()
instance_picking_container = st.empty()
def write_instance_picker():
    format_func = (lambda x: x) if not annotate else \
        lambda x: ' '.join(x.split()[:2]) + (
        '' if len(comp_anns) == 0 or f'{split} {x}' not in set(
            comp_anns.instance) else
        ' ({})'.format(', '.join([
            f'"{ann}"' if ann != annotator_name else 'You' for ann in set(
            comp_anns[comp_anns.instance == f'{split} {x}'].annotator)]))
        )
    instances = list(df.iloc[valid_instances].instance_name)
    last_instance_index = 0 if 'last_instance' not in st.session_state.keys() \
        else instances.index(st.session_state['last_instance'])
    instance_name = st.selectbox(
        f'Instances ({num_valid})',
        instances,
        index=last_instance_index,
        key='instance selection',
        on_change=reset_episode_state,
        format_func=format_func)
    return instance_name
with instance_picking_container:
    instance_name = write_instance_picker()
# if annotate and (len(comp_anns) == 0 or annotator_name not in set(
#         comp_anns[comp_anns.instance == f'{split} {instance_name}'].annotator)):
#     with instance_picking_warning:
#         st.warning("You have not submitted this instance yet. Don't change "
#                    "the instance until you have submitted the annotations!")
if annotate and len(comp_anns) > 0 and \
        f'{split} {instance_name}' in set(comp_anns.instance):
    rows = comp_anns[comp_anns.instance == f'{split} {instance_name}']
    comp_num_reports = set(rows.num_reports)
    assert len(comp_num_reports) == 1
    comp_num_reports = next(iter(comp_num_reports))
    comp_model_and_sorting = {
        (i, row.annotator): set(
            [(k[0], v[0]) for k, v in row.items()
             if isinstance(k, tuple) and k[1] == 'sort_by_model_order'
             and v == v])
        for i, row in rows.iterrows()}
    comp_model_and_sorting = set().union(*comp_model_and_sorting.values())
    if len(comp_model_and_sorting) >= 3:
        st.warning(
            'All instances are already annotated on this instance: {}'
            .format(', '.join([str(x) for x in comp_model_and_sorting])))
        comp_model_and_sorting = None
else:
    comp_num_reports = None
    comp_model_and_sorting = None
if 'episode' in st.session_state.keys() and \
        instance_name != st.session_state['episode']['instance']:
    reset_episode_state()
if not annotate and control_state:
    st.button('Reset session', key='reset session',
              on_click=reset_episode_state)
    jump_timestep_container = st.container()
if num_valid == 0:
    st.warning('No example to show')
    st.stop()
# TODO: info exists in metadata, compute class prevelances
# if os.path.exists('class_prevelances.pkl'):
#     with open('class_prevelances.pkl', 'rb') as f:
#         class_prevelances = pkl.load(f)
# else:
#     class_prevelances = None
#     st.write('To allow sorting evidence based on the difference with class '
#              'prevelance, you need to postprocess the instance metadata in the'
#              ' sidebar.')

# st.write(str(env.model.model.lm_head.weight.device))
instance_index = np.where(df.instance_name == instance_name)[0].item()
if 'episode' not in st.session_state.keys():
    st.session_state['episode'] = {'instance': instance_name}
    # store the return value of reset for the current instance
    with st.spinner(
            'Extracting information from reports to set up environment...'):
        st.session_state['episode']['reset'] = env.reset(
            options=get_reset_options(args, instance_index))
        if use_random_start_idx:
            info = st.session_state['episode']['reset'][1]
            if comp_num_reports is None:
                assert min_num_reports - 1 <= info['max_timesteps'] // 2 - 1
                random_start_idx = random.randint(
                    info['current_report'] + min_num_reports - 1,
                    info['current_report'] + info['max_timesteps'] // 2 - 1)
            else:
                random_start_idx = info['current_report'] + \
                    comp_num_reports - 1
            st.session_state['episode']['reset'] = env.reset(
                options=get_reset_options(
                    args, instance_index, start_report_index=random_start_idx))
    # store the actions for the current instance
    st.session_state['episode']['actions'] = {}
    # store the return value of the step function for the current instance
    st.session_state['episode']['steps'] = {}
    st.session_state['episode']['skip_to'] = None
    st.session_state['episode']['num_evidence_revealed'] = {}


if annotate and annotator_name == '':
    st.warning(
        'You need to specify an annotator name to submit annotations.')
    st.stop()

container = st.empty()

observation, info = st.session_state['episode']['reset']
if not annotate and control_state:
    with jump_timestep_container:
        value = len(st.session_state['episode']['steps']) + \
            1 if st.session_state['episode']['skip_to'] is None else \
            st.session_state['episode']['skip_to']
        jump_to = st.number_input(
            'Jump to a step', min_value=1, max_value=info['max_timesteps'],
            value=value)
        st.button('Jump', on_click=JumpTo(args, env, jump_to, instance_index))
reward = 0
i = 0
terminated, truncated = False, False


def display_state(observation, info, reward):
    terminated = not info['is_valid_timestep']
    if not annotate:
        st.subheader("Information")
        st.write(f'**timestep**: {i + 1}')
        st.write('**report**: {}'.format(info['current_report'] + 1))
        st.write(
            f'**is_terminated**: {terminated}')
        st.write(
            '**evidence_is_retrieved**: {}'.format(
                observation['evidence_is_retrieved']))
    with st.expander('Environment secrets'):
        st.write('**reward for previous action**: {}'.format(reward))
        cumulative_reward = sum(
            [reward for _, reward, _, _, _ in st.session_state[
                'episode']['steps'].values()])
        st.write('**cumulative reward**: {}'.format(cumulative_reward))
        st.write('**max_future_reports**: {}'.format(
            info['max_timesteps'] // 2))
        for k, v in info.items():
            if k in ['current_report', 'past_reports', 'future_reports',
                     'current_targets', 'max_timesteps']:
                continue
            st.write('**{}**: {}'.format(k, v))
    if terminated:
        st.warning(
            'Not a valid timestep! '
            'You can move to another instance.')


def evidence_annotations(
        step_index, evidence_index, sort_by, actor_checkpoint,
        selected_option_strings=None):
    if selected_option_strings is None:
        selected_option_strings = ['overall']
    max_cols = 3
    if len(selected_option_strings) > max_cols:
        cols = []
        for row_idx in range((len(selected_option_strings) // max_cols) + 1):
            cols += list(st.columns(max_cols))
    else:
        cols = st.columns(len(selected_option_strings))
    anns = {}
    evidence_is_useful = False
    for col, option in zip(cols, selected_option_strings):
        option = option.split(' (')[0]
        with col:
            if option != 'overall':
                st.write(f'##### {option}')
            relevance = st.radio(
                f'Is the evidence relevant to {option}?',
                ['0 - Not Useful',
                '1 - Weak Correlation',
                '2 - Useful',
                '3 - Very Useful'],
                key=f'relevant {instance_name} {actor_checkpoint} {split} {step_index} {evidence_index} {sort_by} {option}')
            impact = None
            if not relevance.startswith('0'):
                evidence_is_useful = True
                impact = st.radio(
                    f'Does the predicted impact of the evidence on the likelihood of {option} align with intuition?',
                    ['Yes', 'No'],
                    key=f'individual impact {instance_name} {actor_checkpoint} {split} {step_index} {evidence_index} {sort_by} {option}')
            anns[option] = {
                'relevance': relevance, 'impact': impact}
    if evidence_is_useful:
        evidence_was_seen = st.radio(
            'Did you see this evidence in your initial assesment of the patient?',
            ['Yes', 'No'],
            key=f'has been seen {instance_name} {actor_checkpoint} {split} {step_index} {evidence_index} {sort_by}')
        anns['evidence_was_seen'] = evidence_was_seen
    notes = st.text_area(
        'Other notes (if needed)',
        key=f'notes {instance_name} {actor_checkpoint} {split} {step_index} {evidence_index} {sort_by}')
    anns['notes'] = notes
    return anns


def get_static_bias_dist(actor, parameter_votes_info):
    new_param_votes_info = {}
    for k, v in parameter_votes_info.items():
        if k == 'context_embeddings':
            v = v[:1]
        elif k == 'param_votes':
            v = v[:, :1]
            # with torch.no_grad():
            #     v = actor.observation_embedder.batch_embed(
            #         actor.observation_embedder.context_encoder, ['no evidence found'])
        elif k == 'context_info':
            # this makes it so that the evidence is ignored and only the
            # bias is used
            v = ['no evidence']
            # pass
        new_param_votes_info[k] = v
    with torch.no_grad():
        param_info = actor.votes_to_parameters(new_param_votes_info)
        meta_dist = actor.parameters_to_dist(*param_info['params'])
        return actor.get_mean([meta_dist])[0]


def get_evidence_dist(actor, parameter_votes_info, evidence_idx):
    if actor.has_bias and actor.config.static_diagnoses is None and \
            evidence_idx > 0:
        # first piece of context is always the bias vector
        slc = torch.tensor([0, evidence_idx])
    else:
        slc = torch.tensor([evidence_idx])
    new_param_votes_info = {}
    for k, v in parameter_votes_info.items():
        if k == 'context_embeddings':
            v = v[slc]
        elif k == 'param_votes':
            v = v[:, slc]
        new_param_votes_info[k] = v
    with torch.no_grad():
        param_info = actor.votes_to_parameters(new_param_votes_info)
        meta_dist = actor.parameters_to_dist(*param_info['params'])
        return actor.get_mean([meta_dist])[0]


def get_evidence_distributions(actor, parameter_votes_info):
    evidence_distributions = []
    for evidence_idx in range(parameter_votes_info['param_votes'].shape[1]):
        evidence_distributions.append(get_evidence_dist(
            actor, parameter_votes_info, evidence_idx))
    return evidence_distributions


def get_evidence_scores(
        actor, actor_checkpoint, action, options, selected_options, parameter_votes_info, parameters_info, evidence_distributions, i):
    option_indices = torch.tensor(selected_options)
    sort_options = []
    if 'context_attn_weights' in parameters_info.keys():
        sort_options += ['Sort by attention']
    if reward_type in ['continuous_dependent', 'ranking']:
        sort_options += ['Sort by the denominator of the softmax']
    if len(selected_options) >= 2:
        if actor.has_bias:
            sort_options += [
                'Sort by MSE with the bias logits',
                'Sort by the symmetric KL divergence with the bias distribution',
                'Sort by the symmetric KL divergence with the agg/bias distribution',
            ]
        # peakiness options
        sort_options += [
            'Sort by the entropy of the induced distributions',
            'Sort by the maximum probability',
        ]
    if not actor.config.use_raw_sentences:
        sort_options += ['LLM Confidence']
    for selected_option in selected_options:
        sort_options += [f'Sort by the probability of \"{options[selected_option]}\"']
    if annotate:
        if 'annotation_sort_options' not in st.session_state['episode'].keys():
            sort_options = [sort_options[0]]
            if not actor.config.use_raw_sentences:
                sort_options += ['LLM Confidence']
            if comp_model_and_sorting is not None:
                for model, sorting in comp_model_and_sorting:
                    if model == actor_checkpoint:
                        sort_options.remove(sorting)
            if len(sort_options) > 1:
                random.shuffle(sort_options)
            sort_options = sort_options[:1]
            print('Sorting Method:', sort_options[0])
            st.session_state['episode']['annotation_sort_options'] = \
                sort_options
        sorting_methods = st.session_state[
            'episode']['annotation_sort_options']
    else:
        sorting_methods = []
        if pick_how_to_sort_evidence:
            # st.write('Choose how to sort the evidence.')
            with st.expander('Choose how to sort the evidence'):
                for j, x in enumerate(sort_options):
                    if st.checkbox(
                            x,
                            value=j==0,
                            key=f'sort by {i} {x} {actor_checkpoint}'):
                        sorting_methods.append(x)
        else:
            sorting_methods.append(sort_options[0])
    if actor.has_bias:
        if actor.config.static_diagnoses is not None:
            bias_logits = get_static_bias_dist(
                actor, parameter_votes_info)[option_indices]
        else:
            bias_logits = evidence_distributions[0][option_indices]
        agg_over_bias_logits = torch.tensor(action[option_indices]) - \
            bias_logits
    sort_info = []
    for sort_by in sorting_methods:
        if sort_by.startswith('Sort by the probability of'):
            option_to_sort_by = options.index(sort_by.split('\"')[1])
            selected_option_to_sort_by = selected_options.index(option_to_sort_by)
            # params = param_votes[option_to_sort_by].transpose(0, 1)
            # meta_dist = actor.parameters_to_dist(*params)
            # scores = actor.get_mean([meta_dist])[0]
            scores = []
            for dist in evidence_distributions:
                if reward_type == 'continuous_independent':
                    scores.append(torch.sigmoid(
                        dist[option_indices])[selected_option_to_sort_by])
                else:
                    scores.append(torch.softmax(
                        dist[option_indices], 0)[selected_option_to_sort_by])
            ascending = False
            score_format = lambda x: 'Prob={:.1f}%'.format(x * 100)
        elif sort_by == 'Sort by attention':
            # option x evidence
            attn = parameters_info['context_attn_weights']
            scores = attn[option_indices].mean(0)
            ascending = False
            score_format = lambda x: 'Attn={:.1f}%'.format(x * 100)
        elif sort_by == 'Sort by the denominator of the softmax':
            scores = []
            for dist in evidence_distributions:
                scores.append(torch.exp(dist).sum())
            ascending = False
            score_format = lambda x: 'Softmax denom={:.1f}'.format(x)
        elif sort_by == 'Sort by the entropy of the induced distributions':
            scores = []
            # TODO: potentially do something different here
            #   if env_kwargs['reward_type'] == 'continuous_independent'
            for dist in evidence_distributions:
                entropy = Categorical(
                    torch.softmax(dist[option_indices], 0)).entropy()
                scores.append(entropy)
            scores = np.array(scores)
            ascending = True
            score_format = lambda x: 'Entropy={:.1f}'.format(x)
        elif sort_by == 'Sort by the maximum probability':
            scores = []
            for dist in evidence_distributions:
                if reward_type == 'continuous_independent':
                    scores.append(torch.sigmoid(dist[option_indices]).max())
                else:
                    scores.append(torch.softmax(dist[option_indices], 0).max())
            scores = np.array(scores)
            ascending = False
            score_format = lambda x: 'Max Prob={:.1f}%'.format(x * 100)
        elif sort_by == 'Sort by MSE with the bias logits':
            scores = []
            for dist in evidence_distributions:
                scores.append(nn.MSELoss()(
                    dist[option_indices], bias_logits).item())
            scores = np.array(scores)
            ascending = False
            score_format = lambda x: 'MSE with Bias Logits={:.1f}'.format(x)
        elif sort_by == 'Sort by the symmetric KL divergence with the bias ' \
                'distribution':
            scores = []
            if reward_type == 'continuous_independent':
                bias_distribution = Bernoulli(
                    torch.sigmoid(bias_logits))
            else:
                bias_distribution = Categorical(
                    torch.softmax(bias_logits, 0))
            for dist in evidence_distributions:
                if reward_type == 'continuous_independent':
                    dist = Bernoulli(
                        torch.sigmoid(dist[option_indices]))
                else:
                    dist = Categorical(
                        torch.softmax(dist[option_indices], 0))
                scores.append((
                    kl_divergence(dist, bias_distribution) +
                    kl_divergence(bias_distribution, dist)).mean() / 2)
            scores = np.array(scores)
            ascending = False
            score_format = lambda x: 'Sym KL with Bias={:.1f}'.format(x)
        elif sort_by == 'Sort by the symmetric KL divergence with the agg/bias ' \
                'distribution':
            scores = []
            if reward_type == 'continuous_independent':
                agg_over_bias_distribution = Bernoulli(
                    torch.sigmoid(agg_over_bias_logits))
            else:
                agg_over_bias_distribution = Categorical(
                    torch.softmax(agg_over_bias_logits, 0))
            for dist in evidence_distributions:
                if reward_type == 'continuous_independent':
                    dist = Bernoulli(
                        torch.sigmoid(dist[option_indices]))
                else:
                    dist = Categorical(
                        torch.softmax(dist[option_indices], 0))
                scores.append((
                    kl_divergence(dist, agg_over_bias_distribution) +
                    kl_divergence(agg_over_bias_distribution, dist)).mean() / 2)
            scores = np.array(scores)
            ascending = True
            score_format = lambda x: 'Sym KL with Agg/Bias={:.2f}'.format(x)
        elif sort_by == 'LLM Confidence':
            scores = []
            for ci in parameter_votes_info['context_info']:
                if ci.startswith('evidence'):
                    confidence_str = ci.split(
                        ' (')[1][:-1].split(', ')[1].split()[1]
                    score = -1 if confidence_str == 'unk' else \
                        float(confidence_str)
                else:
                    score = -1
                scores.append(score)
            scores = np.array(scores)
            ascending = False
            score_format = lambda x: 'LLM Confidence={:.2f}'.format(x)
        else:
            raise Exception
        sort_info.append((sort_by, ascending, scores, score_format))
    return sort_info


def get_evidence_strings(parameter_votes_info, parameters_info, scores, score_format):
    context_strings = parameter_votes_info['context_strings']
    context_info = parameter_votes_info['context_info']
    new_strings = []
    for cs, ci, score in zip(context_strings, context_info, scores):
        evidence = '[report]' if ci == 'report' else cs
        score_string = score_format(score)
        new_strings.append({
            'score': score_string.split('=')[1],
            'evidence': evidence,
        })
        if ci.startswith('evidence'):
            split_evidence = evidence.split(':')
            query = split_evidence[0]
            the_rest = ':'.join(split_evidence[1:])
            new_evidence, day = the_rest.split(' (day ')
            new_strings[-1]['evidence'] = new_evidence.strip()
            new_strings[-1]['query'] = query
            new_strings[-1]['day'] = int(float(day[:-1])) \
                if day[:-1] != 'nan' else 'UNK'
            report_num_str = ci.split(' (')[1][:-1].split(', ')[0]
            new_strings[-1]['report_number'] = report_num_str.split()[1]
        elif ci.startswith('sentence'):
            new_evidence, day = evidence.split(' (day ')
            new_strings[-1]['evidence'] = new_evidence.strip()
            new_strings[-1]['day'] = int(float(day[:-1])) \
                if day[:-1] != 'nan' else 'UNK'
            report_num_str = ci.split(' (')[1][:-1]
            new_strings[-1]['report_number'] = report_num_str.split()[1]
    return new_strings


def make_evidence_plot(options, option_dist, bias_dist=None, show_bias=False):
    option_dist = torch.sigmoid(option_dist) \
        if reward_type == 'continuous_independent' else \
        torch.softmax(option_dist, 0)
    options = [o.split('(')[0].strip() for o in options]
    if bias_dist is not None:
        fig, axs = plt.subplots(
            3 if show_bias else 2, 1,
            figsize=(3, 7 if show_bias else 5), sharex=True)
        ax0 = axs[0]
        ax0.title.set_text('Prediction')
        ax1 = axs[1]
        ax1.title.set_text('Prediction/Bias')
        if show_bias:
            ax2 = axs[2]
            ax2.title.set_text('Bias')
    else:
        ax0 = None
    make_plot(options, option_dist, ax=ax0)
    if bias_dist is not None:
        bias_dist = torch.sigmoid(bias_dist) \
            if reward_type == 'continuous_independent' else \
            torch.softmax(bias_dist, 0)
        make_plot(options, option_dist / bias_dist, not_dist=True,
                  ylabel='Risk over Baseline', ax=ax1)
        ax1.plot([-.5, -.5 + len(options)], [1.0, 1.0], '--', color='black')
        ax1.set_xlim([-.5,  -.5 + len(options)])
        if show_bias:
            make_plot(options, bias_dist, ax=ax2)
        st.pyplot(fig)


def display_evidence_string(es):
    # needs (background, key, value)
    tag_html_template = \
        '<span><span style="display:inline-flex;flex-direction:row;align-ite' \
        'ms:center;background:{};border-radius:0.5rem;padding:0.25rem 0.5rem' \
        ';overflow:hidden;line-height:1"><span style="margin-right:0.5rem;fo' \
        'nt-size:0.75rem;opacity:0.5">{}</span><span style="border-right:1px' \
        ' solid;opacity:0.1;margin-right:0.5rem;align-self:stretch"></span>{' \
        '}</span></span>'
    score = es['score']
    evidence = es['evidence']
    tag_info = []
    if not annotate:
        tag_info += [('#803df533', 'SCORE', score)]
        if 'query' in es.keys():
            tag_info += [('#803df533', 'QUERY', es['query'])]
    if 'report_number' in es:
        tag_info += [
            ('#803df533', 'REPORT NUMBER', es['report_number']),
            ('#803df533', 'DAY', es['day']),
        ]
    st.write(
        ' '.join([
            tag_html_template.format(background, key, value)
            for background, key, value in tag_info
        ]),
        unsafe_allow_html=True
    )
    st.write('#### ' + evidence.replace('\n', '\n#### '))


class ShowMoreEvidence:
    def __init__(self, i, sort_by):
        self.i = i
        self.sort_by = sort_by
    def __call__(self):
        st.session_state['episode']['num_evidence_revealed'][
            (self.i, self.sort_by)] += num_evidence_at_a_time


def display_action_evidence(
        args, actor_checkpoint, i, action, parameter_votes_info, parameters_info,
        options, selected_options=None):
    actor = get_actor(args, actor_checkpoint)
    if isinstance(actor.observation_embedder, BertObservationEmbedder):
        st.write("Not an Interpretable model, nothing to display.")
        return {}
    if selected_options is not None and len(selected_options) > 0:
        selected_options = sorted(selected_options)
        selected_option_strings = [options[idx] for idx in selected_options]
    else:
        st.warning('Some options must be selected to order by.')
        return {}
    with torch.no_grad():
        evidence_dists = get_evidence_distributions(
            actor, parameter_votes_info)
        sort_info = get_evidence_scores(
            actor, actor_checkpoint, action, options, selected_options, parameter_votes_info,
            parameters_info, evidence_dists, i)
    if annotate:
        anns = {'evidence_anns': {}, 'options': options,
                'selected_options': selected_options,
                'sort_by_model_order': [x[0] for x in sort_info]}
        tab_names = [f'Sorting Model {i+1}' for i in range(len(sort_info))]
    else:
        tab_names = [x[0] for x in sort_info]
    tabs = st.tabs(tab_names)
    for (sort_by, ascending, scores, score_format), tab in zip(
            sort_info, tabs):
        with tab:
            evidence_strings = get_evidence_strings(
                parameter_votes_info, parameters_info, scores, score_format)
            if actor.has_bias:
                if actor.config.static_diagnoses is not None:
                    bias_dist = get_static_bias_dist(
                        actor, parameter_votes_info)
                else:
                    assert evidence_strings[0]['evidence'] == '[bias vector]'
                    bias_dist = evidence_dists[0]
            else:
                bias_dist = None
            evidence_info = sorted(zip(
                range(len(evidence_strings)),
                evidence_strings,
                evidence_dists,
                scores), key=lambda x: x[-1] if ascending else -x[-1])
            evidence_info_to_show = evidence_info
            ignore_evidence_types = set([])
            if ignore_selected_evidence:
                st.write('Ignore evidence of:')
                evidence_types = sorted(list(set(
                    [es['query']
                    for _, es, _, _ in evidence_info_to_show
                    if 'query' in es.keys()])))
                num_columns = 3
                columns = st.columns(num_columns)
                for idx, x in enumerate(evidence_types):
                    with columns[idx % num_columns]:
                        if st.checkbox(x):
                            ignore_evidence_types.add(x)
                evidence_info_to_show = [
                    x for x in evidence_info_to_show
                    if 'query' not in x[1].keys() or
                    x[1]['query'] not in ignore_evidence_types]
            if annotate:
                anns['evidence_anns'][sort_by] = {}
            else:
                anns = {}
            if (i, sort_by) not in st.session_state[
                    'episode']['num_evidence_revealed'].keys():
                st.session_state[
                    'episode']['num_evidence_revealed'][(i, sort_by)] = num_evidence_at_a_time
            num_evidence_revealed = st.session_state[
                'episode']['num_evidence_revealed'][(i, sort_by)]
            evidence_info_to_show = evidence_info_to_show[
                :num_evidence_revealed]
            if annotate:
                st.write('Annotate the following evidence snippets retrieved '
                            'by the model. If more evidence is needed to '
                            'adequately predict a diagnosis for a patient, please'
                            ' press the \"Show More Evidence\" button at the '
                            f'bottom. A maximum of {num_evidence_to_annotate} '
                            'evidence snippets can be shown per patient.')
            st.write(f'Only showing the top **{num_evidence_revealed} of '
                        f'{len(evidence_info)}** pieces of evidence.')
            for evidence_number, (j, es, ed, _) in enumerate(
                    evidence_info_to_show):
                st.write(f'### {evidence_number+1}.')
                c1, c2 = st.columns([1, 3])
                with c1:
                    make_evidence_plot(
                        selected_option_strings,
                        ed[torch.tensor(selected_options)],
                        bias_dist=bias_dist[torch.tensor(selected_options)]
                            if bias_dist is not None else None)
                with c2:
                    display_evidence_string(es)
                    if annotate:
                        anns['evidence_anns'][sort_by][j] = \
                            evidence_annotations(
                                i, j, sort_by, actor_checkpoint,
                                selected_option_strings=
                                    selected_option_strings)
                        anns['evidence_anns'][sort_by][j]['evidence'] = es
                        anns['evidence_anns'][sort_by][j]['sorted_idx'] = \
                            evidence_number
                        anns['evidence_anns'][
                            sort_by][j]['evidence_distribution'] = ed.tolist()
            if show_remaining_anns and \
                    len(evidence_info) > num_evidence_revealed:
                with st.expander('The rest of the evidence.'):
                    for evidence_number, (j, es, ed, _) in enumerate(
                            evidence_info[num_evidence_revealed:]):
                        st.write(f'### {evidence_number+1}.')
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            make_evidence_plot(
                                selected_option_strings,
                                ed[torch.tensor(selected_options)],
                                bias_dist=bias_dist[
                                    torch.tensor(selected_options)]
                                    if bias_dist is not None else None)
                        with c2:
                            display_evidence_string(es)
            else:
                st.write(f'{len(evidence_info[num_evidence_revealed:])} '
                            'more evidence snippets not shown.')
                max_evidence = min(
                    num_evidence_to_annotate, len(evidence_info)) \
                    if annotate else len(evidence_info)
                if num_evidence_revealed < max_evidence:
                    st.button(
                        'Show More Evidence',
                        key=f'show more evidence {actor_checkpoint} {i} {sort_by}',
                        on_click=ShowMoreEvidence(i, sort_by))
    return anns


def annotate_timepoint(i, options):
    timepoint_anns = {}
    st.write('Select any illnesses with which the patient has already been '
             'confidently diagnosed:')
    seen_targets = set()
    for option in options:
        if st.checkbox(option, key=f'already seen {instance_name} {split} {option} {i}'):
            seen_targets.add(option)
    timepoint_anns['seen_targets'] = seen_targets
    if len(seen_targets) == 0:
        timepoint_anns['option_likelihood_anns'] = {}
        for option in options:
            timepoint_anns['option_likelihood_anns'][option] = st.radio(
                f'What is the likelihood of the patient having {option}?',
                ['Unlikely', 'Somewhat likely', 'Very likely'],
                key=f'likelihood ann {instance_name} {split} {i} {option}')
    else:
        timepoint_anns['invalid_instance_notes'] = st.text_area('Other notes (if needed)', key=f'invalid instance notes {instance_name} {split} {i}')
    # st.write('Given the notes and evidence seen so far, which of the following are correct diagnoses?')
    # selected_options = set()
    # diagnosis_salience = {}
    # for option in options:
    #     if st.checkbox(option, key=f'correct {option} {i}'):
    #         selected_options.add(option)
    #         diagnosis_salience[option] = st.radio('Select the Salience of the diagnosis.', ['Primary', 'Secondary'], key=f'diagnosis salience {i} {option}')
    # temporal_mistakes = selected_options.intersection(info['current_targets'])
    # extraction_mistakes = selected_options.difference(info['current_targets'])
    # confident_diagnoses_not_in_differentials = st.text_area('Please write a comma-separated list of any confident diagnoses that are not listed.', key=f'text1 {i}')
    # confident_diagnoses_not_in_differentials = [x.strip() for x in confident_diagnoses_not_in_differentials.split(',')]
    # missed_differentials = st.text_area('Please write a comma-separated list of any differential diagnoses that are not listed but should be considered.', key=f'text2 {i}')
    # missed_differentials = [x.strip() for x in missed_differentials.split(',')]
    # if len(temporal_mistakes) > 0:
    #     st.radio(f'The following were not extracted as \"confident\" until later in time: {temporal_mistakes}. Is this because the clinician may have made a mistake?', ['Yes', 'No'], key=f'temporal mistake {i}')
    # if len(extraction_mistakes) > 0:
    #     st.write(f'The following were not extracted as confident diagnoses at all: {extraction_mistakes}. These represent mistakes by the confident diagnosis extraction model.')
    return timepoint_anns


def annotate_timepoint_end(i, options, anns, actor_checkpoint):
    timepoint_anns = {}
    timepoint_anns['option_likelihood_anns2'] = {}
    st.write("Review your previous answers to the following questions. If you feel your opinion has changed, change your answers accordingly.")
    answers = ['Unlikely', 'Somewhat likely', 'Very likely']
    for option in options:
        timepoint_anns['option_likelihood_anns2'][option] = st.radio(
            f'What is the likelihood of the patient having {option}?',
            answers,
            index=answers.index(anns['option_likelihood_anns'][option]),
            key=f'likelihood ann 2 {instance_name} {actor_checkpoint} {split} {i} {option}')
    timepoint_anns['concluding_notes'] = st.text_area('Other notes (if needed)', key=f'concluding notes {instance_name} {actor_checkpoint} {split} {i}')
    # timepoint_anns['prediction_ann'] = st.radio('Does the prediction align with your expert intuition?', ['Yes', 'No'], key=f'prediction ann {i}')
    # timepoint_anns['change_opinion'] = st.radio('Does the evidence change your opinion?', ['Yes', 'No'], key=f'change opinion ann {i}')
    return timepoint_anns


def display_action(args, options, action, action_stddev, i, actor_checkpoint, parameter_votes_info):
    c1, c2 = st.columns([2, 5])
    anns = {}
    with c2: 
        if reward_type == 'continuous_independent':
            ratings = torch.sigmoid(torch.tensor(action))
            st.write('This model\'s scores are interpreted as independent. '
                    '(Probabilities do not sum to 1.)')
        else:
            ratings = torch.softmax(torch.tensor(action), 0).numpy()
            st.write('This model\'s scores are interpreted as not independent. '
                    '(Probabilities sum to 1.)')
        options_ratings_indices = list(
            zip(options, ratings, action_stddev, range(len(action))))
        options_ratings_indices = sorted(
            options_ratings_indices, key=lambda x: -x[1])
        if annotate:
            selected_options = set([
                j for _, _, _, j in options_ratings_indices])
            anns['prediction_aligns'] = st.radio(
                'Does this prediction align with your likelihood intuition.',
                ['Yes', 'No'], key=f'prediction aligns {instance_name} {actor_checkpoint} {split} {i}')
        else:
            selected_options = set()
            for o, r, a_s, j in options_ratings_indices:
                option_string ='({:.1f}%{}) {}'.format(
                    r * 100, '±{:.1f}'.format(a_s * 100) if a_s != 0 else '', o)
                if st.checkbox(option_string, value=True,
                        key=f'option checkbox {instance_name} {actor_checkpoint} {split} {i} {option_string}'):
                    selected_options.add(j)
            st.write('Select options above to use to sort the evidence.')
    with c1:
        actor = get_actor(args, actor_checkpoint)
        if actor.has_bias:
            if actor.config.static_diagnoses is not None:
                bias_dist = get_static_bias_dist(actor, parameter_votes_info)
            else:
                bias_dist = get_evidence_dist(actor, parameter_votes_info, 0)
            bias_dist = bias_dist[torch.tensor(sorted(selected_options))]
        else:
            bias_dist = None
        make_evidence_plot(
            [options[j] for j in selected_options],
            torch.tensor(action)[torch.tensor(sorted(selected_options))],
            bias_dist=bias_dist, show_bias=True)
    return selected_options, anns


def write_annotations(
        split, instance_name, i, observation, info, action, reports, anns):
    all_anns = {
        'info': {
            k: v for k, v in info.items()
            if k not in ['past_reports', 'future_reports']},
        'obs': {
            k: v for k, v in observation.items()
            if k not in ['past_reports', 'reports', 'evidence']},
        'action': action.tolist(),
        'timestep': i,
        'num_reports': len(reports),
        'instance': f'{split} {instance_name}',
        'time_for_initial_assesment':
            st.session_state['episode']['time_for_initial_assesment'],
        'visited_reports':
            st.session_state['episode']['report_tracker'].visited_reports,
        'visited_report_timestamps':
            st.session_state['episode'][
                'report_tracker'].visited_report_timestamps,
        **anns
    }
    # st.success(all_anns)
    if not os.path.exists(args['annotations']['path']):
        os.mkdir(args['annotations']['path'])
    if not os.path.exists(os.path.join(args['annotations']['path'], split)):
        os.mkdir(os.path.join(args['annotations']['path'], split))
    ann_path = os.path.join(
        args['annotations']['path'], split, '_'.join(annotator_name.split()))
    if not os.path.exists(ann_path):
        os.mkdir(ann_path)
    next_idx = 0
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            next_idx = max(
                next_idx, int(filename.split('.')[0].split('_')[1]) + 1)
    new_filepath = os.path.join(ann_path, f'ann_{next_idx}.pkl')
    # st.success(f'Writing to: {new_filepath}')
    print(f'Writing to: {new_filepath}')
    with open(new_filepath, 'wb') as f:
        pkl.dump({next_idx: all_anns}, f)


class SubmitAnnotationsCallback:
    def __init__(self, split, instance_name, i, observation, info, action, reports, anns):
        self.split = split
        self.instance_name = instance_name
        self.i = i
        self.observation = observation
        self.info = info
        self.action = action
        self.reports = reports
        self.anns = anns
    def __call__(self):
        write_annotations(
            self.split, self.instance_name, self.i, self.observation,
            self.info, self.action, self.reports, self.anns)
        st.session_state['last_instance'] = self.instance_name


# def get_current_annotations():
#     ann_path = os.path.join(
#         args['annotations']['path'], split, '_'.join(annotator_name.split()))
#     return get_annotations(ann_path)


import time
class Timer:
    def __init__(self, num_seconds):
        self.num_total_seconds = num_seconds
        self.time_started = time.time()

    def get_seconds(self):
        return self.num_total_seconds - (time.time() - self.time_started)


class ReportTracker:
    def __init__(self):
        self.visited_report_timestamps = []
        self.visited_reports = []

    def record_current_report(self, timer, report_name):
        self.visited_report_timestamps.append(-timer.get_seconds())
        self.visited_reports.append(report_name)


def display_report(reports_df, key):
    if len(reports_df) == 0:
        st.write('No reports to display.')
        return
    report_names = reports_df.report_name
    def write_report(key_addon, timer=None, tracker=None):
        report_name = st.selectbox(
            'Choose a report', report_names,
            key=key + ' ' + key_addon, index=0)
        if timer is not None and tracker is not None:
            tracker.record_current_report(timer, report_name)
        report_row = reports_df[reports_df.report_name == report_name].iloc[0]
        st.write(f'Description: {report_row.description}')
        st.divider()
        st.write(report_row.text)
        # tabs = st.tabs([x[:10] for x in report_names])
        # for tab, report_name in zip(tabs, report_names):
        #     report_row = reports_df[reports_df.report_name == report_name].iloc[0]
        #     with tab:
        #         st.write(f'Description: {report_row.description}')
        #         st.divider()
        #         st.write(report_row.text)
    timer = None
    tracker = None
    if annotate:
        if 'display_report_timer' in st.session_state['episode'].keys() and \
                st.session_state['episode']['display_report_timer'] is None:
            # st.write("Reports have disappeared!")
            with st.expander("Reports"):
                write_report('1')
            st.warning('{:.0f} seconds'.format(
                st.session_state['episode']['time_for_initial_assesment']))
            return
        if 'display_report_timer' not in st.session_state['episode'].keys():
            start_button_container = st.empty()
            if start_button_container.button('Start Timer for Initial Assesment'):
                start_button_container.empty()
                st.session_state['episode']['display_report_timer'] = Timer(0)
                st.session_state['episode']['report_tracker'] = ReportTracker()
            else:
                st.stop()
        timer = st.session_state['episode']['display_report_timer']
        tracker = st.session_state['episode']['report_tracker']
    container = st.empty()
    with container.container():
        write_report('2', timer=timer, tracker=tracker)
    if annotate:
        assert timer is not None
        timer_context = st.empty()
        button_context = st.empty()
        if not button_context.button('Done with reports'):
            with timer_context:
                while timer is not None:
                    time.sleep(timer.get_seconds() % 1)
                    seconds = timer.get_seconds()
                    # if seconds_left < 0:
                    #     seconds_left = 0
                    # st.warning('{:.0f} seconds left!'.format(seconds_left))
                    st.warning('{:.0f} seconds'.format(-seconds))
                    # if seconds <= 0:
                    #     st.session_state['episode']['display_report_timer'] = None
        else:
            st.session_state['episode'][
                'time_for_initial_assesment'] = -timer.get_seconds()
            st.session_state['episode']['display_report_timer'] = None
        assert 'time_for_initial_assesment' in st.session_state['episode'].keys()
        with timer_context:
            st.warning('{:.0f} seconds'.format(
                st.session_state['episode']['time_for_initial_assesment']))
        button_context.empty()
        container.empty()
        # container.write("Reports have disappeared!")
        with container.expander("Reports"):
            write_report('3')


def raw_timestep(
        annotate, show_state, show_evidence, i, info, observation, reward,
        display_state, display_report, show):
    reports = pd.concat(
                [df_from_string(observation['past_reports']),
                    df_from_string(observation['reports'])]).reset_index()
    reports['date'] = pd.to_datetime(reports['date'])
    reports = process_reports(
                reports, reference_row_idx=info['start_report'])
    reports = reports[::-1]
    if show:
        if show_state:
            display_state(observation, info, reward)
        st.subheader("Reports")
        display_report(
                    reports,
                    f'timestep {i + 1}')
        if not annotate and show_evidence:
            st.subheader("Evidence")
            if observation['evidence'].strip() != '':
                st.write(df_from_string(observation['evidence']))
            else:
                st.write('No evidence yet!')
    options = df_from_string(observation['options']).apply(
        lambda r: f'{r.option} ({r.type})', axis=1).to_list()
    if show and annotate:
        st.write('### Part 1: Pre-model Annotations')
        anns.update(annotate_timepoint(i, options))
        # if not annotate or len(anns['seen_targets']) == 0:
        #     st.subheader("Prediction")
    return reports, options


def show_model_outputs(
        args, annotate, use_actor, reward_type, env, i,
        observation, show, anns, actor_checkpoint, options):
    last_tab_context_for_submit_button = None
    if show:
        if annotate:
            if len(anns['seen_targets']) == 0:
                prediction_container, evidence_container, concluding_container = st.tabs([
                    'Part 2A: Model Prediction',
                    'Part 2B: Evidence',
                    'Part 2C: Concluding Thoughts'
                ])
        else:
            prediction_container, evidence_container = st.tabs(['Model Prediction', 'Evidence'])
    if not use_actor:
        action = [0] * len(options)
        if show:
            with prediction_container:
                st.write('Rank the options.')
                action = np.array(action, dtype=float)
                action_df = pd.DataFrame(
                            {'rating': {d: a for d, a in zip(options, action)}})
                edited_df = st.experimental_data_editor(
                            action_df, key=f'action editor {i}')
                        # edited_df = st.data_editor(action_df, key=f'action editor {i}')
                action_dict = edited_df.to_dict()['rating']
                action = [action_dict[d] for d in options]
    elif args['random_query'] and not observation['evidence_is_retrieved']:
        action = env.action_space.sample()
    else:
        checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
        assert reward_type == checkpoint_args.env.value.reward_type
        if 'model_anns' not in anns.keys():
            anns['model_anns'] = {}
        anns['model_anns'][actor_checkpoint] = {}
        parameter_votes_info, parameters_info, action, \
                    action_stddev = sample_action(
                    args, observation, actor_checkpoint)
        action = np.array(action, dtype=float)
        if show and (not annotate or len(anns['seen_targets']) == 0):
            with prediction_container:
                        # if not annotate:
                selected_options, action_anns = display_action(
                    args, options, action, action_stddev, i,
                    actor_checkpoint, parameter_votes_info)
                anns['model_anns'][actor_checkpoint].update(action_anns)
                        # else:
                        #     selected_options = list(range(len(options)))
    if show:
        if actor_checkpoint != 'Choose your own actions' and (
                        not annotate or len(anns['seen_targets']) == 0):
            with evidence_container:
                anns['model_anns'][actor_checkpoint].update(
                    display_action_evidence(
                        args, actor_checkpoint, i, action, parameter_votes_info,
                        parameters_info, options,
                        selected_options=selected_options))
        if annotate:
            if len(anns['seen_targets']) == 0:
                with concluding_container:
                    anns['model_anns'][actor_checkpoint].update(
                        annotate_timepoint_end(
                            i, options, anns, actor_checkpoint))
                last_tab_context_for_submit_button = concluding_container
            action_submitted = False
        else:
            action_submitted = st.button(
                'Next Timestep', key=f'submit {i} {actor_checkpoint}')
    else:
        action_submitted = False
    skip_past = st.session_state['episode']['skip_to'] is not None and \
                len(st.session_state['episode']['steps']) + \
                1 < st.session_state['episode']['skip_to']
    return action, action_submitted, skip_past, \
        last_tab_context_for_submit_button


from collections import Counter
def run_timestep(
        args, annotate, use_actor, reward_type, split, env, i,
        instance_name, info, observation, show, anns,
        actor_checkpoints, reward):
    if not show:
        st.write('')
    reports, options = raw_timestep(
        annotate, show_state, show_evidence, i, info, observation, reward,
        display_state, display_report, show)
    if show and annotate and len(anns['seen_targets']) == 0:
        st.write('### Part 2: Model Annotations')
        model_annotations = st.expander('Model Annotations', expanded=False)
    else:
        model_annotations = st.empty()
    with model_annotations:
        if annotate:
            if 'chosen_actor' not in st.session_state['episode'].keys():
                weights=[1] * len(actor_checkpoints)
                counts = Counter([] if comp_model_and_sorting is None else
                    [model for model, _ in comp_model_and_sorting])
                weights[actor_checkpoints.index('all_sentences')] = \
                    1 - counts['all_sentences']
                weights[actor_checkpoints.index('llm_evidence')] = \
                    2 - counts['llm_evidence']
                st.session_state['episode']['chosen_actor'] = random.choices(
                    actor_checkpoints, weights=weights, k=1)[0]
                print('Model:', st.session_state['episode']['chosen_actor'])
            tab_actor_checkpoints = [
                st.session_state['episode']['chosen_actor']]
            tabs = st.tabs([
                f'Prediction Model {idx+1}'
                for idx in range(len(tab_actor_checkpoints))])
        else:
            tab_actor_checkpoints = actor_checkpoints
            tabs = st.tabs(tab_actor_checkpoints)
    for idx, (actor_checkpoint, tab) in enumerate(
            zip(tab_actor_checkpoints, tabs)):
        with tab:
            outs = show_model_outputs(
                args, annotate, use_actor, reward_type, env, i,
                observation, show, anns, actor_checkpoint, options)
            if idx == 0:
                action, action_submitted, skip_past, _ = outs
    last_tab_context_for_submit_button = outs[3]
    if show and annotate:
        if len(anns['seen_targets']) > 0:
            model_annotations.empty()
        def make_submit_button():
            return st.button(
                'Submit Annotations',
                key=f'submit anns {instance_name} {i}',
                on_click=SubmitAnnotationsCallback(
                    split, instance_name, i, observation, info, action,
                    reports, anns),
            )
        if last_tab_context_for_submit_button is not None:
            with last_tab_context_for_submit_button:
                submit_annotations = make_submit_button()
        else:
            submit_annotations = make_submit_button()
        if submit_annotations:
            # write_annotations(
            #     split, instance_name, i, observation, info, action, reports, anns)
            st.success('The target diagnoses were: ' + ', '.join(
                        [f'\"{k}\" in {v} reports'
                            for k, v in info['target_countdown'].items()]))
            with instance_picking_warning:
                st.success("You have submitted this instance!")
        elif len(comp_anns) == 0 or annotator_name not in set(
                comp_anns[comp_anns.instance == f'{split} {instance_name}'
                          ].annotator):
            with instance_picking_warning:
                st.warning(
                    "You have not submitted this instance yet. Don't change "
                    "the instance until you have submitted the annotations!")
        st.expander('Annotations').write(
            get_complementary_annotations(args, split))
        # st.expander('Annotations').write(get_current_annotations())
    return action, action_submitted, skip_past


if use_actor:
    actor_checkpoints = list(args['models'].keys())
else:
    actor_checkpoints = ['Choose your own actions']
num_steps_visualized = 0
while not (terminated or truncated):
    if i not in st.session_state['episode']['steps'].keys():
        if num_steps_visualized > 0:
            container.empty()
            sleep(0.01)
        num_steps_visualized += 1
        show = observation['evidence_is_retrieved'] or not args['skip_query']
        anns = {}
        with container.container():
            action, action_submitted, skip_past = run_timestep(
                args, annotate, use_actor, reward_type, split, env, i,
                instance_name, info, observation, show, anns,
                actor_checkpoints, reward)
        if not action_submitted and not skip_past and not (
                not observation['evidence_is_retrieved'] and args['skip_query']):
            st.session_state['episode']['skip_to'] = None
            st.stop()
        with st.spinner('Taking step'):
            st.session_state['episode']['actions'][i] = action
            st.session_state['episode']['steps'][i] = env.step(action)
    observation, reward, terminated, truncated, info = st.session_state['episode']['steps'][i]
    i += 1
container.empty()
with container.container():
    display_state(observation, info, reward)
    st.write('Done! Environment was {}.'.format(
        'terminated' if terminated else 'truncated'))
