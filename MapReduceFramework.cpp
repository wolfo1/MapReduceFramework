#include "MapReduceFramework.h"
#include "Barrier.h"
#include <atomic>
#include <vector>
#include <algorithm>
#include <iostream>
#include <pthread.h>

#define MUTEX_ERR "System Error: Mutex error."
#define PTHREAD_ERR_CREATE "System Error: pthread_create error."
#define PTHREAD_ERR_JOIN "System Error: pthread_join error."
#define ALLOC_ERR "System Error: Failed to allocate memory."

enum mutexAction {INIT, LOCK, UNLOCK, DESTROY};

void doMutexAction(pthread_mutex_t &mutex, mutexAction action)
{
  int errFlag = 0;
  switch(action)
    {
      case INIT:
        errFlag = pthread_mutex_init(&mutex, NULL);
        break;
      case LOCK:
        errFlag = pthread_mutex_lock(&mutex);
        break;
      case UNLOCK:
        errFlag = pthread_mutex_unlock(&mutex);
        break;
      case DESTROY:
        errFlag = pthread_mutex_destroy (&mutex);
        break;
    }
    if (errFlag != 0)
    {
      std::cerr << MUTEX_ERR << std::endl;
      exit(EXIT_FAILURE);
    }
}

class JobContext;

struct ThreadContext{
    int tid;
    IntermediateVec* map_output = new IntermediateVec;
    JobContext *job;
    ThreadContext(int tid, JobContext *job) :
    tid(tid), job(job) {}
};

struct JobContext
    {
    const MapReduceClient *client;
    const InputVec *inputVec;
    OutputVec *outputVec;
    int threadsNum;
    JobState jobState;
    // all threads
    pthread_t *threads;
    ThreadContext **contexts;
    // counters for state calculations
    std::atomic<uint64_t> counter;
    uint64_t totalToShuffle;
    uint64_t totalToReduce;
    std::vector<IntermediateVec*> shuffleOutput;
    // Mutexes
    pthread_mutex_t outputVecMutex{};
    pthread_mutex_t jobMutex{};
    std::vector<pthread_mutex_t> mutexes;
    // Barriers
    Barrier *reduceBarrier;
    Barrier *shuffleBarrier;
    bool joinedFlag;
    // C'tor
    JobContext(const MapReduceClient *client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel)
    : counter(0), jobState{MAP_STAGE, 0},
      totalToShuffle(0), mutexes(multiThreadLevel - 1), joinedFlag(false), totalToReduce(0)
    {
      this->client = client;
      this->inputVec = &inputVec;
      this->outputVec = &outputVec;
      this->threadsNum = multiThreadLevel;
      // Init threads and thread contexts arrays
      this->threads = new pthread_t[multiThreadLevel];
      this->contexts = new ThreadContext* [multiThreadLevel];
      // init barriers
      this->reduceBarrier = new Barrier(multiThreadLevel);
      this->shuffleBarrier = new Barrier(multiThreadLevel);
      if (!threads || !contexts || !reduceBarrier || !shuffleBarrier)
      {
        std::cerr << ALLOC_ERR << std::endl;
        exit(EXIT_FAILURE);
      }
      // init mutexes
      doMutexAction(this->outputVecMutex, INIT);
      doMutexAction(this->jobMutex, INIT);
      for (int i = 0; i < (multiThreadLevel - 1); ++i)
      {
          doMutexAction(this->mutexes.at(i), INIT);
        }
    }
    // d'tor
    ~JobContext()
    {
      delete[] this->threads;
      for (int i = 0; i < this->threadsNum; ++i)
      {
        delete this->contexts[i];
      }
      delete[] this->contexts;
      for (int j = 0; j < this->threadsNum - 1; ++j)
      {
        doMutexAction(this->mutexes.at(j), DESTROY);
      }
      doMutexAction(this->outputVecMutex, DESTROY);
      doMutexAction(this->jobMutex, DESTROY);
      delete this->reduceBarrier;
      delete this->shuffleBarrier;
    }
    };

void threadMap(void *context)
{
    auto *tc = (ThreadContext *) context;
    while (tc->job->counter < tc->job->inputVec->size())
    {
        doMutexAction(tc->job->mutexes.at(tc->tid), LOCK);
        uint64_t curr_count = (tc->job->counter)++;
        if (curr_count < tc->job->inputVec->size())
        {
            InputPair p = tc->job->inputVec->at(curr_count);
            tc->job->client->map(p.first, p.second, tc);
        }
        doMutexAction(tc->job->mutexes.at(tc->tid), UNLOCK);
    }
}

void shuffle(ThreadContext* tc){
    std::vector<IntermediateVec*> all_vectors;
    // combining all vectors
    for (int i = 0; i < tc->job->threadsNum; ++ i) {
        // get number of items to shuffle (to be used in getState)
        all_vectors.push_back(tc->job->contexts[i]->map_output);
    }
    while (true) {
        //find max key
        K2* max_key = nullptr;
        for(IntermediateVec* vec: all_vectors) {
            if (! vec->empty()) {
                if (!max_key)
                    max_key = vec->back().first;
                else if (!(*vec->back().first < *max_key))
                    max_key = vec->back().first;
            }
        }
        // if all vectors are empty, break
        if (!max_key)
            break;
        //find all instances of key
        auto new_vec = new (std::nothrow)IntermediateVec();
        if (! new_vec){
            std::cerr<< ALLOC_ERR <<std::endl;
            exit(EXIT_FAILURE);
        }
        for(IntermediateVec* vec: all_vectors) {
            while (! vec->empty() && !(*vec->back().first < *max_key)) {
                new_vec->push_back(vec->back());
                vec->pop_back();
                tc->job->counter ++;
            }
        }
        tc->job->shuffleOutput.push_back(new_vec);
    }
  // number of unique keys (value to be used in getState).
  tc->job->totalToReduce = tc->job->shuffleOutput.size();
}

void threadReduce(void *context)
{
  auto *tc = (ThreadContext *) context;
  doMutexAction(tc->job->outputVecMutex, LOCK);
  while(!tc->job->shuffleOutput.empty())
  {
    IntermediateVec *vec = tc->job->shuffleOutput.back();
    tc->job->shuffleOutput.pop_back();
    tc->job->client->reduce (vec, tc->job);
    tc->job->counter++;
  }
  doMutexAction(tc->job->outputVecMutex, UNLOCK);
}

/**
 * cmp function for std::sort to sort Intermediate pairs.
 * @param a pair A
 * @param b pair B
 * @return true if A<B, false otherwise.
 */
bool cmp(const IntermediatePair &a, const IntermediatePair &b) { return *(a.first) < *(b.first); }

void *runThread(void *arg)
{
  auto *tc = (ThreadContext *) arg;
  // map phase
  threadMap(tc);
  // sort phase
  std::sort(tc->map_output->begin(), tc->map_output->end(), cmp);
  tc->job->shuffleBarrier->barrier();
  // shuffle phase
  if (tc->tid == 0)
  {
    tc->job->jobState.stage = SHUFFLE_STAGE;
    tc->job->totalToShuffle = tc->job->counter;
    tc->job->counter.store(0);
    shuffle(tc);
    tc->job->jobState.stage = REDUCE_STAGE;
    tc->job->counter.store(0);
  }
  tc->job->reduceBarrier->barrier();
  // reduce phase
  threadReduce(tc);
  return nullptr;
}

void emit2(K2 *key, V2 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    IntermediatePair p = IntermediatePair(key, value);
    tc->map_output->push_back(p);
}

void emit3(K3 *key, V3 *value, void *context)
{
    auto *job = (JobContext *) context;
    OutputPair p = OutputPair(key, value);
    job->outputVec->push_back(p);
}

JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
                            {
    auto *job = new JobContext(&client, inputVec, outputVec, multiThreadLevel);
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        job->contexts[i] = new ThreadContext(i, job);
        if (job->contexts[i] == nullptr)
          {
            std::cerr << ALLOC_ERR << std::endl;
            exit(EXIT_FAILURE);
          }
    }
    for (int j = 0; j < multiThreadLevel; ++j)
    {
        if (pthread_create(job->threads + j, NULL, runThread, job->contexts[j]) != 0)
        {
            std::cerr << PTHREAD_ERR_CREATE << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return (JobHandle) job;
}

void waitForJob(JobHandle job) {
    JobContext *j = (JobContext *) job;
    if (!j->joinedFlag)
    {
      doMutexAction(j->jobMutex, LOCK);
      j->joinedFlag = true;
      for (int i = 0; i < j->threadsNum; i++)
      {
        if (pthread_join (j->threads[i], NULL) != 0)
        {
          std::cerr << PTHREAD_ERR_JOIN << std::endl;
          exit (EXIT_FAILURE);
        }
      }
      doMutexAction(j->jobMutex, UNLOCK);
    }
}

/**
 * get state and completion % of the job.
 * @param job the job to get state of
 * @param state store stage and percentage in this JobState
 */
void getJobState(JobHandle job, JobState *state) {
    auto *j = (JobContext *) job;
    doMutexAction(j->jobMutex, LOCK);
    // find out job state and percentage completed
    switch(j->jobState.stage)
      {
        case(MAP_STAGE):
          state->stage = MAP_STAGE;
          state->percentage =  100.f * (float) j->counter / (float) j->inputVec->size();
          break;
        case(SHUFFLE_STAGE):
          state->stage = SHUFFLE_STAGE;
          state->percentage = 100.f * (float) j->counter / (float) j->totalToShuffle;
          break;
        case(REDUCE_STAGE):
          state->stage = REDUCE_STAGE;
          state->percentage = 100.f * (float) j->counter / (float) j->totalToReduce;
          break;
        case UNDEFINED_STAGE:
          break;
      }
    doMutexAction(j->jobMutex, UNLOCK);
}


 void closeJobHandle(JobHandle job)
 {
     auto *j = (JobContext *) job;
     waitForJob(j);
     delete j;
 }