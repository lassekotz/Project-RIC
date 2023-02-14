def reformat_json_to_txt(from_path, to_path):
    with open(from_path) as f:
        data = f.read()
        list_format = []
        data2 = data.split(',')
        for datapoint in data2:
            list_format.append(datapoint.replace('[', '').replace(']','').replace(' ', '') + '\n')

        list_format[-1] = list_format[-1].replace('\n', '')

        print(list_format)

    with open(to_path, 'w') as f2:
        for item in list_format:
            f2.write(item)



reformat_json_to_txt('label.json', 'labels.txt')