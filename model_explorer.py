import streamlit as st
from utils import get_args
import pandas as pd
import os
import pickle as pkl
from omegaconf import OmegaConf
from ehr_diagnosis_agent.models.actor import InterpretableDirichletActor, \
    InterpretableNormalActor, InterpretableBetaActor, InterpretableDeltaActor
import torch
import matplotlib.pyplot as plt
import seaborn as sns


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
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 20, ha='right')
    for i, v in enumerate(probs):
        ax.text(i - .2, v.item() + 0.01, '{:.1f}%'.format(v.item() * 100))
    if fig is not None:
        st.pyplot(fig)


def make_evidence_plot(options, option_dist, bias_dist=None, show_bias=False):
    option_dist = torch.sigmoid(option_dist) \
        if args['env']['reward_type'] == 'continuous_independent' else \
        torch.softmax(option_dist, 0)
    options = [o.split('(')[0].strip() for o in options]
    if bias_dist is not None:
        fig, axs = plt.subplots(
            1, 3 if show_bias else 2,
            figsize=(10 if show_bias else 7, 2), sharex=True)
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
            if args['env']['reward_type'] == 'continuous_independent' else \
            torch.softmax(bias_dist, 0)
        make_plot(options, option_dist / bias_dist, not_dist=True,
                  ylabel='Risk over Baseline', ax=ax1)
        ax1.plot([-.5, -.5 + len(options)], [1.0, 1.0], '--', color='black')
        ax1.set_xlim([-.5,  -.5 + len(options)])
        if show_bias:
            make_plot(options, bias_dist, ax=ax2)
        st.pyplot(fig)


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


def get_annotations(ann_path):
    anns_df = {}
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            with open(os.path.join(ann_path, filename), 'rb') as f:
                anns_df.update(pkl.load(f))
    anns_df = pd.DataFrame(anns_df).transpose()
    return anns_df


def get_all_annotations(args):
    anns_df = pd.DataFrame([])
    ann_dirs = [args['annotations']['model_explorer_anns_path']] \
        + args['annotations']['model_explorer_anns_complementary']
    for ann_dir in ann_dirs:
        if os.path.exists(ann_dir):
            for annotator_path in os.listdir(ann_dir):
                anns_df_temp = get_annotations(os.path.join(
                    ann_dir, annotator_path))
                anns_df_temp['annotator'] = [annotator_path] * len(anns_df_temp)
                anns_df = pd.concat([anns_df, anns_df_temp])
    return anns_df


def submit_annotations(args, annotator_name, annotations):
    if not os.path.exists(args['annotations']['model_explorer_anns_path']):
        os.mkdir(args['annotations']['model_explorer_anns_path'])
    ann_path = os.path.join(
        args['annotations']['model_explorer_anns_path'],
        '_'.join(annotator_name.split()))
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
        pkl.dump({next_idx: annotations}, f)


class SubmitAnnotations:
    def __init__(self, args, annotator_name, annotations):
        self.args = args
        self.annotator_name = annotator_name
        self.annotations = annotations
    def __call__(self):
        submit_annotations(self.args, self.annotator_name, self.annotations)


st.set_page_config(layout="wide")
st.title('Interpretable Model Explorer')
args = get_args('config.yaml')
actor_checkpoint = st.selectbox(
    'Select your Model', options=sorted(list(args['models'].keys())))
input_text = st.text_area(
    "Write a piece of evidence to see how it would inluence the model's "
    "decision.")
if input_text == "":
    st.warning("You need to write some evidence to continue.")
day = st.number_input(
    "Choose the day in relation to the present that this evidence was seen.",
    max_value=0, value=0)
checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
actor = get_actor(args, actor_checkpoint)
risk_factors_df = pd.read_csv(args['env']['risk_factors_file'], delimiter='\t')
diagnoses = sorted(list(set(risk_factors_df.diagnosis)))
risk_factors = set()
for i, row in risk_factors_df.iterrows():
    risk_factors = risk_factors.union(
        [rf.strip() for rf in row['risk factors'].split(',')])
risk_factors = list(risk_factors)
if actor_checkpoint == "llm_evidence":
    query = st.selectbox(
        'Select the query that you might have used to retrieve this evidence '
        'using an LLM',
        ['Choose a Query'] +
        [(x, 'diagnosis') for x in diagnoses] +
        [(x, 'risk factor') for x in risk_factors],
        format_func=lambda x: f'{x[0]} ({x[1]})' if isinstance(x, tuple) else x)
    if query == "Choose a Query":
        st.warning("You need to specify a query to continue.")
    if query == "Choose a Query" or input_text == "":
        st.stop()
else:
    query = ""
    if input_text == "":
        st.stop()
from datetime import datetime, timedelta
current_date = datetime.today()
evidence_date = current_date - timedelta(days=-day)
observation = {
    "past_reports": pd.DataFrame({
        "date": [evidence_date.strftime('%Y-%m-%d')],
        "text": [input_text]
    }).to_csv(index=False),
    "reports": pd.DataFrame({
        "date": [current_date.strftime('%Y-%m-%d')],
        "text": ["dummy sentence."]
    }).to_csv(index=False),
    "options": pd.DataFrame({
        "option": diagnoses,
        "type": ['diagnosis'] * len(diagnoses),
    }).to_csv(index=False),
    "evidence": pd.DataFrame({
        "day": [day],
        query: [input_text],
    }).to_csv(index=False),
    "evidence_is_retrieved": True,
}
parameter_votes_info, parameters_info, action, action_stddev = sample_action(
    args, observation, actor_checkpoint)
has_dummy = actor_checkpoint == "all_sentences"
if actor.has_bias:
    if actor.config.static_diagnoses is not None:
        bias_dist = get_static_bias_dist(actor, parameter_votes_info)
        assert len(parameter_votes_info['context_strings']) == 1 + has_dummy
        processed_evidence = parameter_votes_info['context_strings'][0]
    else:
        bias_dist = get_evidence_dist(actor, parameter_votes_info, 0)
        assert len(parameter_votes_info['context_strings']) == 2 + has_dummy
        processed_evidence = parameter_votes_info['context_strings'][1]
else:
    bias_dist = None
    assert len(parameter_votes_info['context_strings']) == 1 + has_dummy
    processed_evidence = parameter_votes_info['context_strings'][0]
st.write('Context String:')
st.write(processed_evidence)
make_evidence_plot(
    diagnoses,
    torch.tensor(action),
    bias_dist=torch.tensor(bias_dist) if bias_dist is not None else None,
    show_bias=True)
annotator_name = st.text_input('Annotator Name')
if annotator_name == '':
    st.warning(
        'You need to specify an annotator name to submit annotations.')
    st.stop()
annotations = {
    'annotator': annotator_name,
    'model': actor_checkpoint,
    'input_text': input_text,
    'bias_dist': bias_dist.tolist() if bias_dist is not None else None,
    'predicted_dist': action.tolist(),
    'notes': st.text_area('Write any notes you have about this example.'),
}
submitted = st.button(
    'Submit Annotation',
    on_click=SubmitAnnotations(args, annotator_name, annotations))
if submitted:
    st.success(f'Submitted Annotations: {str(annotations)}')
st.write(get_all_annotations(args))
