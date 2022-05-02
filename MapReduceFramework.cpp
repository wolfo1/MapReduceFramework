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

enum mutexActions {INIT, LOCK, UNLOCK, DESTROY};

// All mutex actions wrapper
void mutexAction(pthread_mutex_t &mutex, mutexActions action)
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

// struct for ThreadContext
struct ThreadContext{
    int tid;
    // the output of the Map phase for this thread
    IntermediateVec* mapOutput = new IntermediateVec;
    JobContext *job;
    ThreadContext(int tid, JobContext *job) :
    tid(tid), job(job) {}
};

// struct for JobContext
struct JobContext
    {
    const MapReduceClient *client;
    const InputVec *inputVec;
    OutputVec *outputVec;
    int threadsNum;
    // all threads
    pthread_t *threads;
    ThreadContext **contexts;
    // counters for state calculations
    std::atomic<uint64_t> counter;
    std::vector<IntermediateVec*> shuffleOutput;
    // Mutexes
    pthread_mutex_t outputVecMutex{};
    pthread_mutex_t jobMutex{};
    pthread_mutex_t atomicMutex{};
    std::vector<pthread_mutex_t> mutexes;
    // Barriers
    Barrier *reduceBarrier;
    Barrier *shuffleBarrier;
    bool joinedFlag;
    // C'tor
    JobContext(const MapReduceClient *client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel)
    : counter(0), mutexes(multiThreadLevel - 1), joinedFlag(false)
    {
      this->client = client;
      this->inputVec = &inputVec;
      this->outputVec = &outputVec;
      this->threadsNum = multiThreadLevel;
      this->threads = new pthread_t[multiThreadLevel];
      this->contexts = new ThreadContext* [multiThreadLevel];
      this->reduceBarrier = new Barrier(multiThreadLevel);
      this->shuffleBarrier = new Barrier(multiThreadLevel);
      if (!threads || !contexts || !reduceBarrier || !shuffleBarrier)
      {
        std::cerr << ALLOC_ERR << std::endl;
        exit(EXIT_FAILURE);
      }
      mutexAction (this->outputVecMutex, INIT);
      mutexAction (this->jobMutex, INIT);
      mutexAction(this->atomicMutex, INIT);
      for (int i = 0; i < (multiThreadLevel - 1); ++i)
        mutexAction (this->mutexes.at (i), INIT);
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
          mutexAction (this->mutexes.at (j), DESTROY);
      }
      mutexAction (this->outputVecMutex, DESTROY);
      mutexAction (this->jobMutex, DESTROY);
      delete this->reduceBarrier;
      delete this->shuffleBarrier;
    }
    };

// code for map stage for every thread
void threadMap(void *context)
{
    auto *tc = (ThreadContext *) context;
    while (((tc->job->counter.load() << 33) >> 33) < tc->job->inputVec->size())
    {
        mutexAction (tc->job->mutexes.at (tc->tid), LOCK);
        // if there are still pair to pull from inputVec, pull them.
        uint64_t curr_count = (tc->job->counter)++;
        curr_count = (curr_count << 33) >> 33;
        if (curr_count < tc->job->inputVec->size())
        {
            InputPair p = tc->job->inputVec->at(curr_count);
            tc->job->client->map(p.first, p.second, tc);
        }
        else
            (tc->job->counter)--;
        mutexAction (tc->job->mutexes.at (tc->tid), UNLOCK);
    }
}

// code for shuffle stage for thread 0
void shuffle(ThreadContext* tc){
    std::vector<IntermediateVec*> all_vectors;
    all_vectors.reserve(tc->job->threadsNum);
    // combining all vectors
    for (int i = 0; i < tc->job->threadsNum; ++ i) {
        // get number of items to shuffle (to be used in getState)
        all_vectors.push_back(tc->job->contexts[i]->mapOutput);
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
                (tc->job->counter++);
            }
        }
        tc->job->shuffleOutput.push_back(new_vec);
    }
}

// code for step Reduce for each thread
void threadReduce(void *context)
{
  auto *tc = (ThreadContext *) context;
  mutexAction (tc->job->outputVecMutex, LOCK);
  while(!tc->job->shuffleOutput.empty())
  {
    IntermediateVec *vec = tc->job->shuffleOutput.back();
    uint64_t vecSize = vec->size();
    tc->job->shuffleOutput.pop_back();
    tc->job->client->reduce (vec, tc->job);
    (tc->job->counter) += vecSize;
  }
  mutexAction (tc->job->outputVecMutex, UNLOCK);
}

/**
 * cmp function for std::sort to sort Intermediate pairs.
 * @param a pair A
 * @param b pair B
 * @return true if A<B, false otherwise.
 */
bool cmp(const IntermediatePair &a, const IntermediatePair &b) { return *(a.first) < *(b.first); }

// the code which every thread runs
void *runThread(void *arg)
{
  auto *tc = (ThreadContext *) arg;
  // map phase
  threadMap(tc);
  // sort phase
  std::sort(tc->mapOutput->begin(), tc->mapOutput->end(), cmp);
  (tc->job->counter) += (tc->mapOutput->size() << 31);
  tc->job->shuffleBarrier->barrier();
  // shuffle phase - Only thread 0 shuffles.
  if (tc->tid == 0)
  {
      uint64_t temp;
      // delete current count, update stage to SHUFFLE
      mutexAction(tc->job->atomicMutex, LOCK);
      temp = (((tc->job->counter) >> 31) << 31) + (1UL << 62);
      (tc->job->counter) = temp;
      mutexAction(tc->job->atomicMutex, UNLOCK);
      shuffle(tc);
      // move current shuffle count to totalToReduce, update stage to REDUCE
      mutexAction(tc->job->atomicMutex, LOCK);
      temp = (((tc->job->counter) << 33) >> 2) + (3UL << 62);
      (tc->job->counter) = temp;
      mutexAction(tc->job->atomicMutex, UNLOCK);
  }
  tc->job->reduceBarrier->barrier();
  // reduce phase
  threadReduce(tc);
  return nullptr;
}

/**
 * takes (K2,V2) and emits an IntermediatePair, pushes it to the mapoutput of
 * the thread
 * @param key K2
 * @param value V2
 * @param context of the thread which called func
 */
void emit2(K2 *key, V2 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    IntermediatePair p = IntermediatePair(key, value);
    tc->mapOutput->push_back(p);
}

/**
 * takes a pair of (K3, V3) and push it to the output vector
 * @param key K3
 * @param value V3
 * @param context the thread which is pushing the pair
 */
void emit3(K3 *key, V3 *value, void *context)
{
    auto *job = (JobContext *) context;
    OutputPair p = OutputPair(key, value);
    job->outputVec->push_back(p);
}

/**
 * start the MapReduce Algorithm.
 * @param client the client which handles the algorithm functions
 * @param inputVec a vector which contains the starting objects
 * @param outputVec the vector to output the final result to
 * @param multiThreadLevel how many threads to use
 * @return the job which will handle the algorithm
 */
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
    job->counter += 1UL << 62;
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

/**
 * wait for job to finish, and join all threads
 * @param job job to wait for
 */
void waitForJob(JobHandle job) {
    auto *j = (JobContext *) job;
    if (!j->joinedFlag)
    {
        mutexAction (j->jobMutex, LOCK);
      j->joinedFlag = true;
      for (int i = 0; i < j->threadsNum; i++)
      {
        if (pthread_join (j->threads[i], NULL) != 0)
        {
          std::cerr << PTHREAD_ERR_JOIN << std::endl;
          exit (EXIT_FAILURE);
        }
      }
        mutexAction (j->jobMutex, UNLOCK);
    }
}

/**
 * get state and completion % of the job.
 * @param job the job to get state of
 * @param state store stage and percentage in this JobState
 */
void getJobState(JobHandle job, JobState *state) {
    auto *j = (JobContext *) job;
    mutexAction(j->atomicMutex, LOCK);
    // find out job state and percentage completed
    uint64_t count = j->counter.load();
    switch(count >> 62)
      {
        case(MAP_STAGE):
          state->stage = MAP_STAGE;
          state->percentage =  100.f * (float) ((count << 33) >> 33) / (float) j->inputVec->size();
          break;
        case(SHUFFLE_STAGE):
          state->stage = SHUFFLE_STAGE;
          state->percentage = 100.f * (float) ((count << 33) >> 33) / (float) ((count << 2) >> 33);
          break;
        case(REDUCE_STAGE):
          state->stage = REDUCE_STAGE;
          state->percentage = 100.f * (float) ((count << 33) >> 33) / (float) ((count << 2) >> 33);
          break;
        case UNDEFINED_STAGE:
            state->stage = UNDEFINED_STAGE;
            state->percentage = 0;
          break;
      }
      mutexAction(j->atomicMutex, UNLOCK);
}

/**
 * close the job and free all memory
 * @param job job to close
 */
 void closeJobHandle(JobHandle job)
 {
     auto *j = (JobContext *) job;
     waitForJob(j);
     delete j;
 }