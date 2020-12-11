import os
import shutil


def init_img_dir(dir_path, img_dir_name='OCR-D-IMG', file_ext='jpg'):
    '''
    Initialize image directory in workspace and move all image files into it.

    Keyword arguments:
    dir_path (str) -- the directory to create the image directory in
    img_dir_name (str) -- the name of the image directory (default: 'OCR-D-IMG')
    file_ext (str) -- the image file extension (default: 'jpg')
    '''
    img_dir = os.path.join(dir_path, img_dir_name)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for img in os.listdir(dir_path):
        if img.endswith('.' + file_ext):
            img_path = os.path.join(dir_path, img)
            img_path_new = os.path.join(img_dir, img)
            print(img_path_new)
            os.rename(img_path, img_path_new)


def init_ocrd_workspace(dir_path):
    '''
    Initialize OCR-D workspace.

    Keyword arguments:
    dir_path (str) -- the directory to initialize the workspace in
    '''
    os.system('ocrd workspace init ' + dir_path)


def add_img_to_mets(dir_path, img_dir_name='OCR-D-IMG', media_type='image/jpg'):
    '''
    Add image files to METS file.

    Keyword arguments:
    dir_path (str) -- the workspace directory
    img_dir_name (str) -- the name of the image directory (default: 'OCR-D-IMG')
    media_type (str) -- the type of the image files (default: 'image/jpg')
    '''
    cwd = os.getcwd()

    img_dir = os.path.join(dir_path, img_dir_name)
    os.chdir(dir_path)
    for img in sorted(os.listdir(img_dir)):
        if img.endswith(media_type.split('/')[-1]):
            img_path = os.path.join(img_dir, img)
            print(img_path)
            page_id = 'P' + img.split('.')[0]
            #file_id = img_dir_name + '_' + img.split('.')[0]
            file_id = img.split('.')[0]
            os.system('ocrd workspace add -g ' + page_id + ' -G ' + img_dir_name + ' -i ' + file_id + ' -m ' + media_type + ' ' + img_path)
    os.chdir(cwd)


def set_ocrd_workspace_identifier(dir_path, id):
    '''
    Set OCR-D workspace identifier.

    Keyword arguments:
    dir_path (str) -- the workspace directory
    id (str) -- the workspace identifier
    '''
    cwd = os.getcwd()

    os.chdir(dir_path)
    os.system('ocrd workspace set-id ' + id)

    os.chdir(cwd)


def run_ocrd_workflow(dir_path, ocrd_workflow_path):
    '''
    Run actual OCR-D workflow.

    Keyword arguments:
    dir_path (str) -- the workspace directory
    ocrd_workflow_path (str) -- the OCR-D workflow directory
    '''
    cwd = os.getcwd()

    run_workflow_path = os.path.join(ocrd_workflow_path, 'run')
    print(run_workflow_path)
    os.chdir(dir_path)
    os.system(run_workflow_path)

    os.chdir(cwd)


def run_ocrd_workflow_parallel(workspaces_path, ocrd_workflow_path):
    '''
    Run actual OCR-D workflow in parallel.

    Keyword arguments:
    workspaces_path (str) -- the directory containing the workspaces
    ocrd_workflow_path (str) -- the OCR-D workflow directory
    '''
    cwd = os.getcwd()
    os.chdir(workspaces_path)

    #for workspace_path in os.listdir(workspaces_path):
    #    workspace_path = os.path.join(workspaces_path, workspace_path)
    #    if not os.path.join(workspace_path, 'mets.xml'):
    #        shutil.rmtree(workspace_path)

    run_workflow_path = os.path.join(ocrd_workflow_path, 'run')
    print(run_workflow_path)
    parallel_workflow_command = 'ls -d -- */ | parallel --eta "cd {} && ' + run_workflow_path + '"'
    os.system(parallel_workflow_command)

    os.chdir(cwd)
