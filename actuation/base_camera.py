class BaseCamera():
    '''
    Base-class for cameras.
    '''

    def __init__(self) -> None:
        pass

    def initialize(self) -> bool:
        '''
        Initializes the camera.
        '''
        raise NotImplementedError("Not implemented.")
        return False

    def shutdown(self) -> bool:
        '''
        Shuts down the video feed.
        '''
        raise NotImplementedError("Not implemented.")
        return False

    def capture_image(self) -> list:
        '''
        Performs capture from the camera.
        '''
        raise NotImplementedError("Not implemented.")
        return []
    