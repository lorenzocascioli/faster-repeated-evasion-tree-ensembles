import time
import os
import pickle
from enum import Enum
from peewee import *

# (!) find a better way
from datasets import * 

CACHE_DIR="" # INSERT CACHE HERE
db = SqliteDatabase(None) # path given at run time

class Attack:
    class Type(Enum):
        Kantchelian = 0
        Veritas = 1
        Cube = 2
        __order__ = 'Kantchelian Veritas Cube'

        def __str__(self):
           return f"{self.name.lower()}"

    class Mode(Enum):
        Full = 0
        Pruned = 1
        Mixed = 2
        __order__ = 'Full Pruned Mixed'

        def __str__(self):
           return f"{self.name.lower()}"

    class Result(Enum):
        SAT = 1
        UNSAT = 0
        UNKNOWN = -1
        SKIPPED = -2
        MISCLASSIFIED = -3

        def __str__(self):
            if self == Attack.Result.SAT:               return "SAT"
            if self == Attack.Result.UNSAT:             return "UNSAT"
            if self == Attack.Result.UNKNOWN:           return "UNKNOWN"
            if self == Attack.Result.SKIPPED:           return "SKIP"
            if self == Attack.Result.MISCLASSIFIED:     return "MISCLASSIFIED"


class MyBaseModel(Model):
    """ abstract class to have common db among all tables """
    class Meta:
        database = db
        #primary_key = CompositeKey('index', 'fold', 'delta', 'mode')# <-- key!

class Result(MyBaseModel):
    """ table to store results """

    # composite primary key
    index = IntegerField()
    fold = IntegerField()
    delta = FloatField()
    mode = CharField()

    hostname = CharField()
    base_label = IntegerField()
    base_example = BlobField()
    result = CharField() 
    time = DoubleField()
    adv_example = BlobField(null=True) # SAT-only
    distance = DoubleField(null=True) # SAT-only
    model_output = DoubleField(null=True) # SAT-only

    class Meta:
        primary_key = CompositeKey('index', 'fold', 'delta', 'mode')# <-- key!

class Metadata(MyBaseModel):
    """ table to store dtrees metadata"""

    # (!) code assumes we work with same params (except fold-delta)
    delta = FloatField()
    fold = IntegerField()
    #num_examples = IntegerField()
    feats_n = IntegerField()
    fnr = FloatField()
    feats_time = FloatField()
    feats_ex = BlobField() # examples used for feat. selection
    fs = BlobField() #test
    tot_feats = IntegerField()
    dname = CharField()
    attack = CharField()
    model = CharField()

    class Meta:
        primary_key = CompositeKey('fold', 'delta')# <-- key!

# All subclasses of MyBaseModel are tables in db (only one for now)
# We retrieve all of them, and will later check that all exist (or create them)
# NOTE: table naming could change in the future, might need some adjustments
#       --> right now table_name = class_name.lower()
TABLES = {t.__name__.lower(): t for t in MyBaseModel.__subclasses__()}


#########################     2. DB Management        #########################

def get_db_path(d, model_type, attack_type, seed, N, feats_n, fnr, in_memory=False, hardening=False):
    if in_memory:   return ":memory:"
    if hardening:   return f"{CACHE_DIR}/fastveritas_hardening_{d.name()}_{model_type}_{attack_type}_seed{seed}_N{N}_fnr{int(fnr*100)}_featsN{feats_n}.db"
    return f"{CACHE_DIR}/fastveritas_{d.name()}_{model_type}_{attack_type}_seed{seed}_N{N}_fnr{int(fnr*100)}_featsN{feats_n}.db"
    #return f"{CACHE_DIR}/fastveritas_{THE_DATASETS[d.name()]}_{model_type}_{attack_type}_seed{seed}_N{N}_fnr{int(fnr*100)}_featsN{feats_n}.db"

def print_key(rec):
    return f"(i-{rec['index']}, f-{rec['fold']}, d-{rec['delta']}, m-{rec['mode']})"

def connect_and_setup_db(d, fold, model_type, attack_type, seed, N, feats_n, \
                            fnr, cache, hardening=False): 
    db_path = get_db_path(d, model_type, attack_type, seed, N, feats_n, fnr, \
                            in_memory=not cache, hardening=hardening)
    time.sleep(fold) # to avoid concurrent init. when running in parallel
    db.init(db_path, pragmas={'journal_mode': 'wal'})
    db.connect()
    print(f"\nConnected to database at {db_path}")

    # if any table does not exist, create it
    for table in TABLES.keys():
        if not db.table_exists(table):
            print(f"Creating '{table}' table.")
            db.create_tables([TABLES[table]])
        else:
            print(f"Table '{table}' loaded.")

def close_db_connection():
    db.close()

def print_all_tables():
    for table in db.get_tables():
        print(f"\nTable: {table}")
        for i, column in enumerate(db.get_columns(table)):
            print(f"\t - col. {i+1} --> ", \
                    f"name: {column.name}, " \
                    f"data type: {column.data_type}, " \
                    f"primary key: {column.primary_key}") 

def empty_all_tables():
    for table in TABLES.keys():
        TABLES[table].delete().execute()
        print(f"Table '{table}' has been emptied.")

def delete_fold_data(fold, pruned_only=False):

    time.sleep(1+10*fold)
    with db.atomic():
    #for table in TABLES.keys():
    
        # do 'result'
        table='result'
        if pruned_only:
            TABLES[table].delete().where(TABLES[table].fold == fold, 
                                        TABLES[table].mode == Attack.Mode.Pruned)\
                                        .execute()
            print(f"Data [pruned/mixed search only] for fold {fold} ",\
                    f"removed from table '{table}'.")
        else:
            TABLES[table].delete().where(TABLES[table].fold == fold).execute()
            print(f"Data for fold {fold} removed from table '{table}'.")

        # do metadata
        table = 'metadata'
        TABLES[table].delete().where(TABLES[table].fold == fold).execute()
        print(f"Data for fold {fold} removed from table '{table}'.")


# (!) deprecated (except for metadata): best do bulk updates when you have all results (way faster)
def update_or_insert_res(res, table='result'):
    try:
        TABLES[table].replace(**res).execute()
        #print(f"\nRecord for example {print_key(res)} created.")
    except IntegrityError as e:
        print(f"\nError creating record for ex. {print_key(res)} in table '{table}': {e}")

def bulk_insert(results, fold, table='result'):
    time.sleep(1+7*fold) # to avoid concurrent accesses running in parallel
    with db.atomic():
        for res in results:
            try:
                TABLES[table].replace(**res).execute()
                #print(f"\nRecord for example {print_key(res)} created.")
            except Exception as e:
                print(res)
                print(e)
            #except IntegrityError as e:
            #    print(f"\nError creating record for ex. {print_key(res)} in table '{table}': {e}")
            # except OperationalError as e:
            #     print(f"{e}: retrying insert")
            #     time.sleep(SEED+5*fold)
            #     TABLES[table].replace(**res).execute()
            #     print(f"insert done")

            

def check_cached_runs(fold, delta, read_cached, table='result'):
    # read once to get indices of all cached experiments (quicker than checking every time)
    done, tot = {t: {} for t in Attack.Mode}, 0
    results = {t: {} for t in Attack.Mode}
    
    if read_cached:
        t = time.time()
        for mode in Attack.Mode:
            query = TABLES[table].select(   TABLES[table].index,\
                                            TABLES[table].result).where(  
                                                TABLES[table].fold==fold,
                                                TABLES[table].delta==delta,
                                                TABLES[table].mode==mode)
            cursor = db.execute(query)
            for i in cursor:    done[mode][i[0]] = i[1] 
            tot += len(done[mode].keys())
        print(f"\nGathered {tot} cached results from table '{table}' in {time.time()-t} seconds.")

    return done

def check_hostname(table='result'):
    distinct_values = TABLES[table].select(TABLES[table].hostname).distinct()
    distinct_values = [entry.hostname for entry in distinct_values]

    if len(distinct_values)>1:
        for i in distinct_values:
            if not i.startswith('pinac'):
                raise RuntimeError(f"You have results belonging to multiple (non-pinac) machines in table '{table}'!")

    if len(distinct_values)==1:

        # allow using different pinacs
        if os.uname().nodename.startswith('pinac') and distinct_values[0].startswith('pinac'):
            pass 

        elif os.uname().nodename != distinct_values[0]:
            raise RuntimeError(f"Cached results in table '{table}' come from a different machine!")


def save_metadata(delta, fold, num_examples, feats_n, fnr, feats_time, feats_ex, fs, tot_feats, dname, attack, model):

    meta = {

            "delta": delta, 
            "fold": fold,
            #"num_examples": num_examples,
            "feats_n": feats_n,
            "fnr": fnr,
            "feats_time": feats_time,
            "feats_ex": pickle.dumps(feats_ex),
            "fs": pickle.dumps(fs),
            "tot_feats": tot_feats,
            "dname": dname,
            "attack": attack,
            "model": model
    }

    update_or_insert_res(meta, table='metadata')
    print("\nmetadata for this run successfully saved.")
    #meta["fs"] = pickle.loads(meta["fs"])
    return meta

def read_metadata(fold, delta):
    tab = TABLES['metadata']
    feats = tab.select(
                        tab.delta,
                        tab.fold,
                        #tab.num_examples, 
                        tab.feats_n,
                        tab.fnr,
                        tab.feats_time,
                        tab.feats_ex,
                        tab.fs,  
                        tab.tot_feats, 
                        tab.dname,
                        tab.attack,
                        tab.model)\
                .where((tab.delta==delta) & (tab.fold==fold))\
                .tuples().execute()
    meta = None
    for row in feats:
        meta = {

                "delta": row[0], 
                "fold": row[1],
                #"num_examples": num_examples,
                "feats_n": row[2],
                "fnr": row[3],
                "feats_time": row[4],
                "feats_ex": pickle.loads(row[5]),
                "fs": pickle.loads(row[6]),
                "tot_feats": row[7],
                "dname": row[8],
                "attack": row[9],
                "model": row[10]
        }
        break # (!) assumes only one row present for (fold, delta)
    return meta


def read_fold_res(fold, delta, table='result'):
    """ used to load stuff, do mixed, save stuff """

    # 1. read full+pruned for each index
    fold_condition = (TABLES[table].fold==fold) 
    delta_condition = (TABLES[table].delta==delta)

    read_all = TABLES[table]\
                .select()\
                .where(fold_condition & delta_condition)\
                .namedtuples().execute()
                # (!) added order_by post-ICML to tackle weird ordering issue
                # .order_by(TABLES[table].index)\

    pruned, full = {}, {}
    for (i, fold, delta, mode, host, label, ex, res, time, adv_ex, dist, out) in read_all:
    #for (i, mode, time, res) in read_all:
        row = {
                "index": i,
                "fold": fold,
                "delta": delta,
                "mode": mode,
                "hostname": host,
                "base_label": label,
                # not loaded as this is only used to re-save stuff for mixed
                "base_example": ex, #pickle.loads(ex) if ex else None,
                "result": res,
                "time": time,
                "adv_example": adv_ex, #pickle.loads(adv_ex) if adv_ex else None,
                "distance": dist, 
                "model_output": out
            }
        if mode==str(Attack.Mode.Pruned):    
            pruned[i] = row
        elif mode==str(Attack.Mode.Full):
            full[i] = row

    return full, pruned