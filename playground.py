from disparity.emd import EMD
from disparity.helpers import Helper

helper = Helper(db_name="WorkerSet100K",
                collection_name="workers", N=50)

workers = helper.get_documents()
attributes = helper.get_attributes(workers)
emd = EMD(workers, attributes, f=[0, 1])

print(emd.metric(emd.unbalanced()))
print(emd.metric(emd.balanced()))

