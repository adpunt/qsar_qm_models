import FLuID as fluid
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pickle
import numpy as np

# Set params
params = {
    
    # experiment details
    'details' : 3,                  # level of detail of the experiment (low=1,medium=2,high=3,full=4)
    
    # datafiles
'training_data_file' : 'hERG_lhasa_training',
    'test_data_file' : 'hERG_lhasa_test',
'transfer_data_file' : 'FLuID_full',
  'fluid_label_file' : 'FLuID_labels',
    
    # data sampling
   'validation_ratio': 0.2,        # ratio validation/training
     'transfer_size' : 50000 ,      # sample for the transfer data (-1 = all)
         'test_size' : -1,          # sample for the test data (-1 = all)
     'training_size' : -1,          # sample for the training data (-1 = all)

    # number of teacher/clusters (kMean)
                 'k' : 8,           # number of clusters (kMean)
     'smooth_factor' : 0.05,        # level of post-clustering mixing to avoid fully biased teachers
    
    # teachers
 'teacher_algorithm' : 'rf',        # algorithm used to build the teacher models
    
    # students
 'federated_student' : 'F8',
      'student_size' : 10000,                                              # size of the student (number of labelled Cronos data used)
      'student_sizes' : [100,250,500, 1000,2500,5000,10000,25000,50000],   # sizes of the student ti study the impact of the size
 'student_algorithm' : 'rf',                                               # default algorithm used to build the student models
      'student_mode' : 'balanced',                                         # default mode used to select the student data 
    
    # random seed for reproductibility
      'random_state' : 42,

    # t-SNE settings
         'tsne_size' : 500,
   'tsne_iterations' : 1000,
    
    # replication level
    'replicate_count' : 3,
    
    # fonts
       'figure_font' : dict(family="Arial",size=14,color="black"),
 'small_figure_font' : dict(family="Arial",size=10,color="black"),

    # colors
'figure_color_scale' : [(0,"red"),(0.2,"orange"), (0.3,'yellow'),(1,'green')],
        'bar_colors' : px.colors.qualitative.Prism,
         'green_map' : plt.get_cmap('Greens')
}


"""
    Create a federated dataset by loading and preparing the data.
"""
def federate_data(distance_metric='tanimoto'):
    # if force_sdf True we force reloading from the SDF files
    # as the transfer data is so large it arrives as a zip and will be unzipped into the sdf before it is used
    force_sdf = False 
    training_full = fluid.load_training_data(params, force_sdf)
    test_full = fluid.load_test_data(training_full, params, force_sdf)
    transfer_full = fluid.load_transfer_data(params, force_sdf)

    # Sample the desired size of data according to the experiment parameters
    training_data, test_data, transfer_data, validation_data = fluid.sample_data(training_full, test_full, transfer_full, params)

    # Use the training data as teacher source space
    # and plot the class distribution in this space
    teacher_space = training_data

    # Cluster the source data into k teacher training sets
    teacher_data = fluid.cluster_data_space(teacher_space, 'Teacher', 'T', params['k'], params['smooth_factor'], params)

    # Observe cluster spaces
    fluid.project_teacher_cluster_space(training_data, params)
    fluid.project_teacher_activity_space(training_data, params)

    # Verify that training, test, and transfer spaces overlap
    fluid.plot_transfer_space(training_data, test_data, transfer_data, params)
    fluid.plot_data_space([(transfer_data, "transfer", 1000), (training_data,"training", 500)], params)
    datasets = [(transfer_data, "transfer", 5000)] +[ (teacher_data[i], "T" + str(i), 100) for i in range(1, 1+params['k'])]
    fluid.plot_data_space(datasets, params)
    datasets = [(transfer_data, "transfer", 1000)] + [ (pd.concat([ teacher_data[i] for i in range(1, 1+params['k'])]), "teachers", 500)]
    fluid.plot_data_space(datasets, params)

    # Building the teacher models
    teacher_models = fluid.build_teacher_models(teacher_data, params)

    # Teachers internal cross-validation
    fluid.cross_validate_teachers(teacher_data, params)

    # Teachers validation on ChEMBL space
    teacher_validation_table, teacher_average_table = fluid.validate_teachers(teacher_models, validation_data, params)

    # Teachers validation on the Preissner space
    teacher_validation_table, teacher_average_table = fluid.validate_teachers(teacher_models, test_data, params)

    # Labelling the FLuID transfer data
    # Each public structure is annotated with a hERG predicted label, expressed as a probability distribution between
    # ACTIVE and INACTIVE classes
    force_annotation = True
    label_table = fluid.annotate_transfer_data(transfer_data, teacher_models, teacher_data, params, force_annotation, distance_metric=distance_metric)

    # Federating the labels
    label_table = fluid.federate_teacher_annotations(label_table, params)
    fluid.plot_annotation_distributions(label_table, 800, 600, params)
    fluid.compute_teacher_probability_distributions(label_table, params)
    fluid.plot_confidence_distributions(label_table, 900,800, params)

    # We now have a full set of non-sensitive transfer data annotated by a federation of k teachers

    # Assuming the current file is in the "preprocessing" folder
    file_name = "label_table_" + distance_metric + ".pickle"
    data_path = os.path.join("..", "data", file_name)

    # Save dataset with federated labels in the "data" folder
    with open(data_path, 'wb') as f:
        pickle.dump(label_table, f)

    # Save dataset with federated labels
    with open('label_table_tanimoto.pickle', 'wb') as f:
        pickle.dump(label_table, f)

federate_data()