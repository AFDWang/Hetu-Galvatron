import re

def print_log(logs_dir, datasets, models, maxseqs, methods):
    for model in models:
        print(model)
        total_time = 0.
        for dataset in datasets:
            print(f'{dataset}')
            values = {}
            print('Maxseq  :', end='')
            for maxseq in maxseqs[model]:
                print('   %dK'%maxseq, end='  ')
            print()
            for method in methods:
                print(method.ljust(8), end=': ')
                values[method] = {}
                for maxseq in maxseqs[model]:
                    filename = f'{model}_{dataset}_{maxseq}k_{method}'
                    with open(f'./{logs_dir}/'+filename+'.log', 'r') as file:
                        line = file.readlines()[-2]
                        match = re.search(r'iterations:\s*(\d+\.\d+|\d+)', line)
                        time_value = float(match.group(1))
                        total_time += time_value
                        print(f'{time_value/40:.2f}', end=', ')
                        values[method][maxseq] = time_value
                print()
            print('Acc'.ljust(8), end=': ')
            for maxseq in maxseqs[model]:
                acc = (values['static'][maxseq]/values['flexSP'][maxseq] -1)*100
                print(f'{acc: .2f}%', end=',')
            print()
        print(time_value/60, 'Hours')
        print()
        
        
logs_dir = 'solver_logs_10.4'
methods = ['static', 'adaptive', 'flexSP']

datasets = ['github_1', 'common_crawl', 'wikipedia_1']
models = ['gpt-7b', 'gpt-13b', 'gpt-30b']
maxseqs = {
    'gpt-7b': [384, 192],
    'gpt-13b': [384, 192],
    'gpt-30b': [384, 192],
}
print_log(logs_dir, datasets, models, maxseqs, methods)

datasets = ['common_crawl']
models = ['gpt-7b']
maxseqs = {
    'gpt-7b': [384, 256, 192, 128, 64],
}
print_log(logs_dir, datasets, models, maxseqs, methods)