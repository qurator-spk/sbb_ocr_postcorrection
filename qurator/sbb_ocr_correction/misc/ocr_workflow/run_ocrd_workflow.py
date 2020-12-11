import io
import os
dir_name = os.path.expanduser('~')
import shutil
import time

from data_handling import get_data_from_git_annex, delete_unnecessary_file_links, split_list_in_n_sized_chunks, extract_tar
from ocrd_workflow import init_img_dir, init_ocrd_workspace, add_img_to_mets, set_ocrd_workspace_identifier, run_ocrd_workflow, run_ocrd_workflow_parallel

if __name__ == '__main__':
    parallel_processing = True
    today = '_2020_02_18'

    root_path = dir_name + '/Qurator/qurator-data/DTA/images'
    ocrd_workflow_path = dir_name + '/Qurator/my_ocrd_workflow'
    data_target_path = dir_name + '/Qurator/used_data/ocr/dta_ocr' + today
    data_readme_path = data_target_path + '/README.md'
    ocrd_workspaces = dir_name + '/Qurator/used_data/ocrd_workspaces/workspaces' + today
    cwd = os.getcwd()
    calamari_target_path = os.path.join(data_target_path, 'calamari_ocr')
    tesseract_target_path = os.path.join(data_target_path, 'tesseract_ocr')

    if not os.path.isdir(calamari_target_path):
        os.makedirs(calamari_target_path)
    if not os.path.isdir(tesseract_target_path):
        os.makedirs(tesseract_target_path)
    if not os.path.isdir(ocrd_workspaces):
        os.mkdir(ocrd_workspaces)
    if not os.path.isfile(data_readme_path):
        with io.open(data_readme_path, mode='w') as f_in:
            f_in.write('# List of Processed Files\n\n')

    filename_chunks = list(split_list_in_n_sized_chunks(os.listdir(root_path), 10))
    sub_chunks = filename_chunks[2:3]
    start_time = time.time()

    for index, filename_chunk in enumerate(sub_chunks):

        os.chdir(root_path)

        #####################################################################
        #                                                                   #
        #  Get data from git annex, copy tar to workspace and delete links  #
        #                                                                   #
        #####################################################################

        for filename in filename_chunk:
            file_path = os.path.join(root_path, filename)

            get_data_from_git_annex(file_path)
            shutil.copy(file_path, ocrd_workspaces)
            delete_unnecessary_file_links(file_path)

        os.chdir(ocrd_workspaces)

        #################################
        #                               #
        #  Extract tar and delete them  #
        #                               #
        #################################

        for filename in filename_chunk:
            new_file_path = os.path.join(ocrd_workspaces, filename)
            extract_tar(new_file_path)
            os.remove(new_file_path)

        os.chdir(cwd)

        #######################################################
        #                                                     #
        #  Run OCRD workflow, save OCR and delete workspaces  #
        #                                                     #
        #######################################################

        for filename in filename_chunk:
            foldername = os.path.splitext(filename)[0]
            workspace_path = os.path.join(ocrd_workspaces, foldername)

            if not os.path.exists(os.path.join(workspace_path, 'OCR-D-IMG')):
                init_img_dir(workspace_path)

            # check: process only short documents
            #img_path = os.path.join(workspace_path, 'OCR-D-IMG')
            #if len(os.listdir(img_path)) > 10:
            #    continue

            init_ocrd_workspace(workspace_path)

            add_img_to_mets(workspace_path)

            set_ocrd_workspace_identifier(workspace_path, id='1234567890')

        # check if parallel processing is needed
        if parallel_processing:
            run_ocrd_workflow_parallel(ocrd_workspaces, ocrd_workflow_path)

        for workspace_path in os.listdir(ocrd_workspaces):
            workspace_name = workspace_path
            workspace_path = os.path.join(ocrd_workspaces, workspace_path)

            # check: process only short documents
            #img_path = os.path.join(workspace_path, 'OCR-D-IMG')
            #if len(os.listdir(img_path)) > 10:
            #    continue

            # if no parallel processing, run ocrd workflow on single workspace
            if not parallel_processing:
                run_ocrd_workflow(workspace_path, ocrd_workflow_path)

            try:
                calamari_path = os.path.join(workspace_path, 'OCR-D-OCR-CALAMARI')
                calamari_doc_target_path = os.path.join(calamari_target_path, workspace_name)
                tesseract_path = os.path.join(workspace_path, 'OCR-D-OCR-TESS')
                tesseract_doc_target_path = os.path.join(tesseract_target_path, workspace_name)

                mets_path = os.path.join(workspace_path, 'mets.xml')

                # save OCR to Calamari and Tesseract paths, respectively
                shutil.copytree(calamari_path, calamari_doc_target_path)
                shutil.copytree(tesseract_path, tesseract_doc_target_path)

                # save METS to Calamari and Tesseract paths, respectively
                os.path.copy(mets_path, calamari_doc_target_path)
                os.path.copy(mets_path, tesseract_doc_target_path)
            except FileNotFoundError as fnfe:
                print(fnfe)

            #shutil.rmtree(workspace_path)

            with io.open(data_readme_path, mode='a') as f_in:
                f_in.write(workspace_name + '\n')

        if index == 0:
            break

    end_time = time.time()
    print('Runtime: {}sec'.format(round(end_time-start_time, 2)))
