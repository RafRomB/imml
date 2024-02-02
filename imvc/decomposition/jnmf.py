
class JNMF(object):
    pass
    # train_Xs = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas"))).fit_transform(
    #     train_Xs)
    # mask = [X.notnull().astype(int) for X in train_Xs]
    # mask = Utils.convert_df_to_r_object(mask)
    # train_Xs = Utils.convert_df_to_r_object(train_Xs)
    # nnTensor = importr("nnTensor")
    # clusters = nnTensor.jNMF(train_Xs, M= mask, J= n_clusters)
    # clusters = np.array(clusters[0]).argmax(axis=1)
