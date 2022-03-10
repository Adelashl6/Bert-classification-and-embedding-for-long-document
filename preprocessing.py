import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Bert Classifier')
    parser.add_argument('--blog_path', type=str, default='./data/blogdomains_Round4_RESULTS.csv',
                                       help='Blog list for labelling the data')

    # parser.add_argument('--bert_model', type=str, default="bert-base-uncased")
    parser.add_argument('--data_path', type=str, default='./data/test_data_preprocessed.csv',
                        help='Blog list for labelling the data')
    parser.add_argument('--save_path', type=str, default='./data/preprocessed_blog_200_domain.csv')
    args = parser.parse_args()

    # load blog and data file
    blog = pd.read_csv(args.blog_path)
    blog.dropna(subset=['disinformation', 'Propaganda'])

    data = pd.read_csv(args.data_path)


    print(data.head(5))

    blog_domain_list = data['blogsite_domain']
    mask = [True if blog_domain in blog['Domain'].tolist() else False for blog_domain in blog_domain_list.values.tolist()]
    print('length of data before preprocessing is:', len(blog_domain_list))

    blog_domain_list = blog_domain_list[mask].tolist()
    ids_list = data['id'][mask].tolist()
    # title_list = data['title'][mask].tolist()
    post_clean_list = data['post_clean'][mask].tolist()
    del data


    code_dict = {'N':0, 'I':1, 'Y':2}

    disinfo_list = []
    propaganda_list = []

    for idx, domain in enumerate(blog_domain_list):
        disinfo = blog[blog['Domain'] == domain]['disinformation'].values[0]
        propaganda = blog[blog['Domain'] == domain]['Pro-Russia'].values[0]

        disinfo_list.append(code_dict[disinfo])
        propaganda_list.append(code_dict[propaganda])

    print("length of data after preprocessing is: ", len(blog_domain_list))
    clean_df = pd.DataFrame({"id": ids_list, "blog_domain": blog_domain_list, # "title":title_list,
                             "post_clean":post_clean_list, "disinformation_code":disinfo_list, "propaganda_code":propaganda_list})

    clean_df.disinformation_code = clean_df.disinformation_code.astype(int)
    clean_df.propaganda_code = clean_df.propaganda_code.astype(int)
    clean_df.drop_duplicates(subset=["id"], inplace=True)
    clean_df.dropna(subset=["post_clean"], inplace=True)
    clean_df.to_csv(args.save_path, index=False)


