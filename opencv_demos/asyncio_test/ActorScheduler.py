from collections import deque


class ActorScheduler:
    def __init__(self):
        self._actors = {}  # Mapping of names to actors
        self._msg_queue = deque()  # Message queue
    
    def new_actor(self, name, actor):
        '''
        Admit a newly started actor to the scheduler and give it a name
        '''
        self._msg_queue.append((actor, None))
        self._actors[name] = actor
    
    def send(self, name, msg):
        '''
        Send a message to a named actor
        '''
        actor = self._actors.get(name)
        if actor:
            self._msg_queue.append((actor, msg))
    
    def run(self):
        '''
        Run as long as there are pending messages.
        '''
        while self._msg_queue:
            actor, msg = self._msg_queue.popleft()
            try:
                print('before actor={}-{}'.format(actor.__name__, msg))
                res = actor.send(msg)
                print('after actor={}-{} res={}'.format(actor.__name__, msg, res))
            except StopIteration:
                pass


# Example use
if __name__ == '__main__':
    def printer():
        msg = ''
        while True:
            print('before printer yield msg={}'.format(msg))
            msg = yield 'printer yield msg={}'.format(msg)
            print('Got:', msg)
    
    def counter(sched):
        while True:
            # Receive the current count
            n = yield 'counter yield'
            if n == 0:
                break
            # Send to the printer task
            sched.send('printer', n)
            # Send the next count to the counter task (recursive)
            
            sched.send('counter', n - 1)
    
    sched = ActorScheduler()
    # Create the initial actors,
    sched.new_actor('printer', printer())
    sched.new_actor('counter', counter(sched))
    
    # Send an initial message to the counter to initiate
    print('sched.send')
    sched.send('counter', 10)
    print('sched.send2')
    sched.run()
