from ScafVAE.app.app_utils import *

df_EGFR, df_HER2 = get_demo_properties(max_num=1000, name='binding')  # get molecules with their properties

generation_path = './output'  # path for saving data
tasks = [
    {
        # task name
        'name': 'EGFR_inhibition',

        # input data
        'data': df_EGFR,

        # task type: classification / regression
        'task_type': 'classification',

        # ML model for surrogate model
        'ML_model': 'RF',

        # property is need to be maximized or minimized: max / min
        'optimization_type': 'max',

        # pseudo weight for this property
        'pseudo_weight': 0.5,
    },
    {
        'name': 'HER2_inhibition',
        'data': df_HER2,
        'task_type': 'classification',
        'ML_model': 'RF',
        'optimization_type': 'max',
        'pseudo_weight': 0.5,
    },
]

base_model, surrogate_model = prepare_data_and_train(generation_path, tasks)
df_output = generate_mol(
    10,  # number of generated molecules
    generation_path, tasks, base_model, surrogate_model,
)
print(df_output)  # a dataframe contrains generated molecules with their properties
