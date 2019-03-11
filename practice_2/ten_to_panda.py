import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

print('---------------------')
print('Data Frame: df2')
print(df2)
print('---------------------')

print('\n---------------------')
print('Types of each column: df2')
print(df2.dtypes)
print('---------------------')

print('\n---------------------')
print('Head: df2')
print(df2.head())
print('---------------------')

print('\n---------------------')
print('Tail: df2')
print(df2.tail(3))
print('---------------------')

print('\n---------------------')
print('Data Frame: df\n', df)
print('---------------------')

print('\n---------------------')
print('Index:\n', df.index)
print('---------------------')

print('\n---------------------')
print('Columns:\n', df.columns)
print('---------------------')

print('\n---------------------')
print('to_numpy() (really .values): df\n', df.values)
print('---------------------')

print('\n---------------------')
print("Describe: df\n", df.describe())
print('---------------------')

print('\n---------------------')
print("Transpose: df\n", df.T)
print('---------------------')

print('\n---------------------')
print('Sort by an axis: df\n', df.sort_index(axis=1, ascending=False))
print('---------------------')

print('\n---------------------')
print('Sort by value: df\n', df.sort_values(by='B'))
print('---------------------')

print('\n________Selection_________')

print('\n---------------------')
print('Select a single column\n', df['A'])
print('---------------------')

print('\n---------------------')
print('Selecting via [], which slices the rows:\n')
print('df[0:3]:\n', df[0:3])
print('\ndf[20130101:20130104]:\n', df['20130102':'20130104'])
print('---------------------')

print('\n---------------------')
print('Cross section using a label: df\n', df.loc[dates[0]])
print('---------------------')

print('\n---------------------')
print('Selecting on a multi-axis by labe: df\n', df.loc[:, ['A', 'B']])
print('---------------------')

print('\n---------------------')
print('Showing label slicing, both endpoints included: df\n', df.loc['20130102':'20130104', ['A', 'B']])
print('---------------------')

print('\n---------------------')
print('Getting a scalar value: df\n', df.loc[dates[0], 'A'])
print('---------------------')

print('\n---------------------')
print('Selecte via the position of the passed integers: df\n', df.iloc[3])
print('---------------------')

print('\n---------------------')
print('Selects by integer slices: df\n', df.iloc[3:5, 0:2])
print('---------------------')

print('\n---------------------')
print('By lists of integer position locations: ([1,2,4] and [0,2]) df\n', df.iloc[[1,2,4], [0,2]])
print('---------------------')

print('\n---------------------')
print('Row slicing explicitly: ([1:3, :]) df:\n', df.iloc[1:3, :])
print('---------------------')

print('\n---------------------')
print('Column slicing explicitly: ([:, 1:3]) df:\n', df.iloc[:, 1:3])
print('---------------------')

print('\n---------------------')
print('A specific value: (1:1) df:\n', df.iloc[1:1])
print('---------------------')

print('\n---------------------')
print('Single columns values to select data: (A>0) df\n', df[df.A >0])
print('---------------------')

print('\n---------------------')
print('Boolean condition is met: (>0) df\n', df[df > 0])
print('---------------------')

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print('New df2:\n', df2)

print('\n---------------------')
print('isin() for filtering: df\n', df2[df2['E'].isin(['two', 'four'])])
print('---------------------')

