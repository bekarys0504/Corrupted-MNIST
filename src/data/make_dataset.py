# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    project_dir = Path(__file__).resolve().parents[2]
    
    # create empty tensors
    train_data_img = torch.tensor([])
    train_data_lbl = torch.tensor([])
    test_data_img = torch.tensor([])
    test_data_lbl = torch.tensor([])

    #combine all train data to one tensor
    for i in range(5):
        filepath = str(project_dir)+'/'+str(input_filepath)+'/train_'+str(i)+'.npz'
        data = np.load(filepath)
        data_img = torch.Tensor(data['images'])
        data_lbl = torch.Tensor(data['labels'])
        data_lbl = data_lbl.type(torch.LongTensor)

        train_data_img = torch.cat((train_data_img, data_img))
        train_data_lbl = torch.cat((train_data_lbl, data_lbl))

    #load test data
    filepath = str(project_dir)+'/'+str(input_filepath)+'/test.npz'
    data = np.load(filepath)
    data_img = torch.Tensor(data['images'])
    data_lbl = torch.Tensor(data['labels'])
    data_lbl = data_lbl.type(torch.LongTensor)

    test_data_img = torch.cat((test_data_img, data_img))
    test_data_lbl = torch.cat((test_data_lbl, data_lbl))

    # save to a file
    torch.save(train_data_img, str(project_dir)+'/'+str(output_filepath) +'/train_img.pt')
    torch.save(train_data_lbl, str(project_dir)+'/'+str(output_filepath)+'/train_lbl.pt')
    torch.save(test_data_img, str(project_dir)+'/'+str(output_filepath)+'/test_img.pt')
    torch.save(test_data_lbl, str(project_dir)+'/'+str(output_filepath)+'/test_lbl.pt')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
