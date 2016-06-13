if __name__ == '__main__':
    """
    """
    ##Loading data
    train_df, test_df = loading_data()
    ##Cleaning the data
    train_df_clean, test_df_clean = prepare_data(train_df, test_df)
    ##Solving classification problems inside each grid cell
    th = 5 #Keep place_ids with more than th samples
    process_grid(train_df_clean, test_df_clean, th, n_cell_x*n_cell_y)